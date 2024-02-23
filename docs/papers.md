# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.14020v1 |
|title| Coercing LLMs to do and reveal (almost) anything |
|authors| Jonas GeipingAlex SteinManli ShuKhalid SaifullahYuxin WenTom Goldstein
|links| http://arxiv.org/abs/2402.14020v1 |
|updated| 2024-02-21 18:59:13 UTC |
|summary| It has recently been shown that adversarial attacks on large language modelsLLMs can jailbreak the model into making harmful statements. In this workwe argue that the spectrum of adversarial attacks on LLMs is much larger thanmerely jailbreaking. We provide a broad overview of possible attack surfacesand attack goals. Based on a series of concrete examples we discusscategorize and systematize attacks that coerce varied unintended behaviorssuch as misdirection model control denial-of-service or data extraction.  We analyze these attacks in controlled experiments and find that many ofthem stem from the practice of pre-training LLMs with coding capabilities aswell as the continued existence of strange glitch tokens in common LLMvocabularies that should be removed for security reasons. |


| Item |Content|
| --- |---|
|idx| 2402.14016v1 |
|title| Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment |
|authors| Vyas RainaAdian LiusieMark Gales
|links| http://arxiv.org/abs/2402.14016v1 |
|updated| 2024-02-21 18:55:20 UTC |
|summary| Large Language Models LLMs are powerful zero-shot assessors and areincreasingly used in real-world situations such as for written exams orbenchmarking systems. Despite this no existing work has analyzed thevulnerability of judge-LLMs against adversaries attempting to manipulateoutputs. This work presents the first study on the adversarial robustness ofassessment LLMs where we search for short universal phrases that when appendedto texts can deceive LLMs to provide high assessment scores. Experiments onSummEval and TopicalChat demonstrate that both LLM-scoring and pairwiseLLM-comparative assessment are vulnerable to simple concatenation attackswhere in particular LLM-scoring is very susceptible and can yield maximumassessment scores irrespective of the input text quality. Interestingly suchattacks are transferable and phrases learned on smaller open-source LLMs can beapplied to larger closed-source models such as GPT3.5. This highlights thepervasive nature of the adversarial vulnerabilities across different judge-LLMsizes families and methods. Our findings raise significant concerns on thereliability of LLMs-as-a-judge methods and underscore the importance ofaddressing vulnerabilities in LLM assessment methods before deployment inhigh-stakes real-world scenarios. |


| Item |Content|
| --- |---|
|idx| 2402.14008v1 |
|title| OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems |
|authors| Chaoqun HeRenjie LuoYuzhuo BaiShengding HuZhen Leng ThaiJunhao ShenJinyi HuXu HanYujie HuangYuxiang ZhangJie LiuLei QiZhiyuan LiuMaosong Sun
|links| http://arxiv.org/abs/2402.14008v1 |
|updated| 2024-02-21 18:49:26 UTC |
|summary| Recent advancements have seen Large Language Models LLMs and LargeMultimodal Models LMMs surpassing general human capabilities in varioustasks approaching the proficiency level of human experts across multipledomains. With traditional benchmarks becoming less challenging for thesemodels new rigorous challenges are essential to gauge their advancedabilities. In this work we present OlympiadBench an Olympiad-level bilingualmultimodal scientific benchmark featuring 8952 problems from Olympiad-levelmathematics and physics competitions including the Chinese college entranceexam. Each problem is detailed with expert-level annotations for step-by-stepreasoning. Evaluating top-tier models on OlympiadBench we implement acomprehensive assessment methodology to accurately evaluate model responses.Notably the best-performing model GPT-4V attains an average score of 17.23on OlympiadBench with a mere 11.28 in physics highlighting the benchmarkrigor and the intricacy of physical reasoning. Our analysis orienting GPT-4Vpoints out prevalent issues with hallucinations knowledge omissions andlogical fallacies. We hope that our challenging benchmark can serve as avaluable resource for helping future AGI research endeavors. |


| Item |Content|
| --- |---|
|idx| 2402.14007v1 |
|title| Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models |
|authors| Zhiwei HeBinglin ZhouHongkun HaoAiwei LiuXing WangZhaopeng TuZhuosheng ZhangRui Wang
|links| http://arxiv.org/abs/2402.14007v1 |
|updated| 2024-02-21 18:48:38 UTC |
|summary| Text watermarking technology aims to tag and identify content produced bylarge language models LLMs to prevent misuse. In this study we introduce theconcept of cross-lingual consistency in text watermarking which assessesthe ability of text watermarks to maintain their effectiveness after beingtranslated into other languages. Preliminary empirical results from two LLMsand three watermarking methods reveal that current text watermarkingtechnologies lack consistency when texts are translated into various languages.Based on this observation we propose a Cross-lingual Watermark Removal AttackCWRA to bypass watermarking by first obtaining a response from an LLM in apivot language which is then translated into the target language. CWRA caneffectively remove watermarks by reducing the Area Under the Curve AUC from0.95 to 0.67 without performance loss. Furthermore we analyze two key factorsthat contribute to the cross-lingual consistency in text watermarking andpropose a defense method that increases the AUC from 0.67 to 0.88 under CWRA. |


| Item |Content|
| --- |---|
|idx| 2402.14002v1 |
|title| Hallucinations or Attention Misdirection? The Path to Strategic Value Extraction in Business Using Large Language Models |
|authors| Aline Ioste
|links| http://arxiv.org/abs/2402.14002v1 |
|updated| 2024-02-21 18:40:24 UTC |
|summary| Large Language Models with transformer architecture have revolutionized thedomain of text generation setting unprecedented benchmarks. Despite theirimpressive capabilities LLMs have been criticized for generating outcomes thatdeviate from factual accuracy or display logical inconsistencies phenomenacommonly referred to as hallucinations. This term however has often beenmisapplied to any results deviating from the instructors expectations whichthis paper defines as attention misdirection rather than true hallucinations.Understanding the distinction between hallucinations and attention misdirectionbecomes increasingly relevant in business contexts where the ramifications ofsuch errors can significantly impact the value extraction from these inherentlypre-trained models. This paper highlights the best practices of the PGIPersona Grouping and Intelligence method a strategic framework thatachieved a remarkable error rate of only 315 percent across 4000 responsesgenerated by GPT in response to a real business challenge. It emphasizes thatby equipping experimentation with knowledge businesses can unlockopportunities for innovation through the use of these natively pre-trainedmodels. This reinforces the notion that strategic application grounded in askilled team can maximize the benefits of emergent technologies such as theLLMs. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.14015v1 |
|title| Corrective Machine Unlearning |
|authors| Shashwat GoelAmeya PrabhuPhilip TorrPonnurangam KumaraguruAmartya Sanyal
|links| http://arxiv.org/abs/2402.14015v1 |
|updated| 2024-02-21 18:54:37 UTC |
|summary| Machine Learning models increasingly face data integrity challenges due tothe use of large-scale training datasets drawn from the internet. We study whatmodel developers can do if they detect that some data was manipulated orincorrect. Such manipulated data can cause adverse effects like vulnerabilityto backdoored samples systematic biases and in general reduced accuracy oncertain input domains. Often all manipulated training samples are not knownand only a small representative subset of the affected data is flagged.  We formalize Corrective Machine Unlearning as the problem of mitigating theimpact of data affected by unknown manipulations on a trained model possiblyknowing only a subset of impacted samples. We demonstrate that the problem ofcorrective unlearning has significantly different requirements from traditionalprivacy-oriented unlearning. We find most existing unlearning methodsincluding the gold-standard retraining-from-scratch require most of themanipulated data to be identified for effective corrective unlearning. Howeverone approach SSD achieves limited success in unlearning adverse effects withjust a small portion of the manipulated samples showing the tractability ofthis setting. We hope our work spurs research towards developing better methodsfor corrective unlearning and offers practitioners a new strategy to handledata integrity challenges arising from web-scale training. |


| Item |Content|
| --- |---|
|idx| 2402.14007v1 |
|title| Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models |
|authors| Zhiwei HeBinglin ZhouHongkun HaoAiwei LiuXing WangZhaopeng TuZhuosheng ZhangRui Wang
|links| http://arxiv.org/abs/2402.14007v1 |
|updated| 2024-02-21 18:48:38 UTC |
|summary| Text watermarking technology aims to tag and identify content produced bylarge language models LLMs to prevent misuse. In this study we introduce theconcept of cross-lingual consistency in text watermarking which assessesthe ability of text watermarks to maintain their effectiveness after beingtranslated into other languages. Preliminary empirical results from two LLMsand three watermarking methods reveal that current text watermarkingtechnologies lack consistency when texts are translated into various languages.Based on this observation we propose a Cross-lingual Watermark Removal AttackCWRA to bypass watermarking by first obtaining a response from an LLM in apivot language which is then translated into the target language. CWRA caneffectively remove watermarks by reducing the Area Under the Curve AUC from0.95 to 0.67 without performance loss. Furthermore we analyze two key factorsthat contribute to the cross-lingual consistency in text watermarking andpropose a defense method that increases the AUC from 0.67 to 0.88 under CWRA. |


| Item |Content|
| --- |---|
|idx| 2402.13979v1 |
|title| The Importance of Architecture Choice in Deep Learning for Climate Applications |
|authors| Simon DrägerMaike Sonnewald
|links| http://arxiv.org/abs/2402.13979v1 |
|updated| 2024-02-21 18:09:04 UTC |
|summary| Machine Learning has become a pervasive tool in climate science applications.However current models fail to address nonstationarity induced byanthropogenic alterations in greenhouse emissions and do not routinely quantifythe uncertainty of proposed projections. In this paper we model the AtlanticMeridional Overturning Circulation AMOC which is of major importance toclimate in Europe and the US East Coast by transporting warm water to theseregions and has the potential for abrupt collapse. We can generate arbitrarilyextreme climate scenarios through arbitrary time scales which we then predictusing neural networks. Our analysis shows that the AMOC is predictable usingneural networks under a diverse set of climate scenarios. Further experimentsreveal that MLPs and Deep Ensembles can learn the physics of the AMOC insteadof imitating its progression through autocorrelation. With quantifieduncertainty an intriguing pattern of spikes before critical points ofcollapse in the AMOC casts doubt on previous analyses that predicted an AMOCcollapse within this century. Our results show that Bayesian Neural Networksperform poorly compared to more dense architectures and care should be takenwhen applying neural networks to nonstationary scenarios such as climateprojections. Further our results highlight that big NN models might havedifficulty in modeling global Earth System dynamics accurately and besuccessfully applied in nonstationary climate scenarios due to the physicsbeing challenging for neural networks to capture. |


| Item |Content|
| --- |---|
|idx| 2402.13945v1 |
|title| Probabilistic Neural Networks (PNNs) for Modeling Aleatoric Uncertainty in Scientific Machine Learning |
|authors| Farhad Pourkamali-AnarakiJamal F. HusseiniScott E. Stapleton
|links| http://arxiv.org/abs/2402.13945v1 |
|updated| 2024-02-21 17:15:47 UTC |
|summary| This paper investigates the use of probabilistic neural networks PNNs tomodel aleatoric uncertainty which refers to the inherent variability in theinput-output relationships of a system often characterized by unequal varianceor heteroscedasticity. Unlike traditional neural networks that producedeterministic outputs PNNs generate probability distributions for the targetvariable allowing the determination of both predicted means and intervals inregression scenarios. Contributions of this paper include the development of aprobabilistic distance metric to optimize PNN architecture and the deploymentof PNNs in controlled data sets as well as a practical material science caseinvolving fiber-reinforced composites. The findings confirm that PNNseffectively model aleatoric uncertainty proving to be more appropriate thanthe commonly employed Gaussian process regression for this purpose.Specifically in a real-world scientific machine learning context PNNs yieldremarkably accurate output mean estimates with R-squared scores approaching0.97 and their predicted intervals exhibit a high correlation coefficient ofnearly 0.80 closely matching observed data intervals. Hence this researchcontributes to the ongoing exploration of leveraging the sophisticatedrepresentational capacity of neural networks to delineate complex input-outputrelationships in scientific problems. |


| Item |Content|
| --- |---|
|idx| 2402.13934v1 |
|title| Do Efficient Transformers Really Save Computation? |
|authors| Kai YangJan AckermannZhenyu HeGuhao FengBohang ZhangYunzhen FengQiwei YeDi HeLiwei Wang
|links| http://arxiv.org/abs/2402.13934v1 |
|updated| 2024-02-21 17:00:56 UTC |
|summary| As transformer-based language models are trained on increasingly largedatasets and with vast numbers of parameters finding more efficientalternatives to the standard Transformer has become very valuable. While manyefficient Transformers and Transformer alternatives have been proposed noneprovide theoretical guarantees that they are a suitable replacement for thestandard Transformer. This makes it challenging to identify when to use aspecific model and what directions to prioritize for further investigation. Inthis paper we aim to understand the capabilities and limitations of efficientTransformers specifically the Sparse Transformer and the Linear Transformer.We focus on their reasoning capability as exhibited by Chain-of-Thought CoTprompts and follow previous works to model them as Dynamic Programming DPproblems. Our results show that while these models are expressive enough tosolve general DP tasks contrary to expectations they require a model sizethat scales with the problem size. Nonetheless we identify a class of DPproblems for which these models can be more efficient than the standardTransformer. We confirm our theoretical results through experiments onrepresentative DP tasks adding to the understanding of efficient Transformerspractical strengths and weaknesses. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.14020v1 |
|title| Coercing LLMs to do and reveal (almost) anything |
|authors| Jonas GeipingAlex SteinManli ShuKhalid SaifullahYuxin WenTom Goldstein
|links| http://arxiv.org/abs/2402.14020v1 |
|updated| 2024-02-21 18:59:13 UTC |
|summary| It has recently been shown that adversarial attacks on large language modelsLLMs can jailbreak the model into making harmful statements. In this workwe argue that the spectrum of adversarial attacks on LLMs is much larger thanmerely jailbreaking. We provide a broad overview of possible attack surfacesand attack goals. Based on a series of concrete examples we discusscategorize and systematize attacks that coerce varied unintended behaviorssuch as misdirection model control denial-of-service or data extraction.  We analyze these attacks in controlled experiments and find that many ofthem stem from the practice of pre-training LLMs with coding capabilities aswell as the continued existence of strange glitch tokens in common LLMvocabularies that should be removed for security reasons. |


| Item |Content|
| --- |---|
|idx| 2402.14017v1 |
|title| D-Flow: Differentiating through Flows for Controlled Generation |
|authors| Heli Ben-HamuOmri PunyItai GatBrian KarrerUriel SingerYaron Lipman
|links| http://arxiv.org/abs/2402.14017v1 |
|updated| 2024-02-21 18:56:03 UTC |
|summary| Taming the generation outcome of state of the art Diffusion and Flow-MatchingFM models without having to re-train a task-specific model unlocks a powerfultool for solving inverse problems conditional generation and controlledgeneration in general. In this work we introduce D-Flow a simple framework forcontrolling the generation process by differentiating through the flowoptimizing for the source noise point. We motivate this framework by our keyobservation stating that for Diffusion/FM models trained with Gaussianprobability paths differentiating through the generation process projectsgradient on the data manifold implicitly injecting the prior into theoptimization process. We validate our framework on linear and non-linearcontrolled generation problems including: image and audio inverse problems andconditional molecule generation reaching state of the art performance acrossall. |


| Item |Content|
| --- |---|
|idx| 2402.14015v1 |
|title| Corrective Machine Unlearning |
|authors| Shashwat GoelAmeya PrabhuPhilip TorrPonnurangam KumaraguruAmartya Sanyal
|links| http://arxiv.org/abs/2402.14015v1 |
|updated| 2024-02-21 18:54:37 UTC |
|summary| Machine Learning models increasingly face data integrity challenges due tothe use of large-scale training datasets drawn from the internet. We study whatmodel developers can do if they detect that some data was manipulated orincorrect. Such manipulated data can cause adverse effects like vulnerabilityto backdoored samples systematic biases and in general reduced accuracy oncertain input domains. Often all manipulated training samples are not knownand only a small representative subset of the affected data is flagged.  We formalize Corrective Machine Unlearning as the problem of mitigating theimpact of data affected by unknown manipulations on a trained model possiblyknowing only a subset of impacted samples. We demonstrate that the problem ofcorrective unlearning has significantly different requirements from traditionalprivacy-oriented unlearning. We find most existing unlearning methodsincluding the gold-standard retraining-from-scratch require most of themanipulated data to be identified for effective corrective unlearning. Howeverone approach SSD achieves limited success in unlearning adverse effects withjust a small portion of the manipulated samples showing the tractability ofthis setting. We hope our work spurs research towards developing better methodsfor corrective unlearning and offers practitioners a new strategy to handledata integrity challenges arising from web-scale training. |


| Item |Content|
| --- |---|
|idx| 2402.14013v1 |
|title| Misalignment, Learning, and Ranking: Harnessing Users Limited Attention |
|authors| Arpit AgarwalRad NiazadehPrathamesh Patil
|links| http://arxiv.org/abs/2402.14013v1 |
|updated| 2024-02-21 18:52:20 UTC |
|summary| In digital health and EdTech recommendation systems face a significantchallenge: users often choose impulsively in ways that conflict with theplatforms long-term payoffs. This misalignment makes it difficult toeffectively learn to rank items as it may hinder exploration of items withgreater long-term payoffs. Our paper tackles this issue by utilizing userslimited attention spans. We propose a model where a platform presents itemswith unknown payoffs to the platform in a ranked list to T users over time.Each user selects an item by first considering a prefix window of these rankeditems and then picking the highest preferred item in that window and theplatform observes its payoff for this item. We study the design of onlinebandit algorithms that obtain vanishing regret against hindsight optimalbenchmarks.  We first consider adversarial window sizes and stochastic iid payoffs. Wedesign an active-elimination-based algorithm that achieves an optimalinstance-dependent regret bound of OlogT by showing matching regretupper and lower bounds. The key idea is using the combinatorial structure ofthe problem to either obtain a large payoff from each item or to explore bygetting a sample from that item. This method systematically narrows down theitem choices to enhance learning efficiency and payoff.  Second we consider adversarial payoffs and stochastic iid window sizes. Westart from the full-information problem of finding the permutation thatmaximizes the expected payoff. By a novel combinatorial argument wecharacterize the polytope of admissible item selection probabilities by apermutation and show it has a polynomial-size representation. Using thisrepresentation we show how standard algorithms for adversarial online linearoptimization in the space of admissible probabilities can be used to obtain apolynomial-time algorithm with OsqrtT regret. |


| Item |Content|
| --- |---|
|idx| 2402.14012v1 |
|title| Chasing Convex Functions with Long-term Constraints |
|authors| Adam LechowiczNicolas ChristiansonBo SunNoman BashirMohammad HajiesmailiAdam WiermanPrashant Shenoy
|links| http://arxiv.org/abs/2402.14012v1 |
|updated| 2024-02-21 18:51:42 UTC |
|summary| We introduce and study a family of online metric problems with long-termconstraints. In these problems an online player makes decisions mathbfx_tin a metric space Xd to simultaneously minimize their hitting costf_tmathbfx_t and switching cost as determined by the metric. Over thetime horizon T the player must satisfy a long-term demand constraintsum_t cmathbfx_t geq 1 where cmathbfx_t denotes the fractionof demand satisfied at time t. Such problems can find a wide array ofapplications to online resource allocation in sustainable energy and computingsystems. We devise optimal competitive and learning-augmented algorithms forspecific instantiations of these problems and further show that our proposedalgorithms perform well in numerical experiments. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.14015v1 |
|title| Corrective Machine Unlearning |
|authors| Shashwat GoelAmeya PrabhuPhilip TorrPonnurangam KumaraguruAmartya Sanyal
|links| http://arxiv.org/abs/2402.14015v1 |
|updated| 2024-02-21 18:54:37 UTC |
|summary| Machine Learning models increasingly face data integrity challenges due tothe use of large-scale training datasets drawn from the internet. We study whatmodel developers can do if they detect that some data was manipulated orincorrect. Such manipulated data can cause adverse effects like vulnerabilityto backdoored samples systematic biases and in general reduced accuracy oncertain input domains. Often all manipulated training samples are not knownand only a small representative subset of the affected data is flagged.  We formalize Corrective Machine Unlearning as the problem of mitigating theimpact of data affected by unknown manipulations on a trained model possiblyknowing only a subset of impacted samples. We demonstrate that the problem ofcorrective unlearning has significantly different requirements from traditionalprivacy-oriented unlearning. We find most existing unlearning methodsincluding the gold-standard retraining-from-scratch require most of themanipulated data to be identified for effective corrective unlearning. Howeverone approach SSD achieves limited success in unlearning adverse effects withjust a small portion of the manipulated samples showing the tractability ofthis setting. We hope our work spurs research towards developing better methodsfor corrective unlearning and offers practitioners a new strategy to handledata integrity challenges arising from web-scale training. |


| Item |Content|
| --- |---|
|idx| 2402.14009v1 |
|title| Geometry-Informed Neural Networks |
|authors| Arturs BerzinsAndreas RadlerSebastian SanokowskiSepp HochreiterJohannes Brandstetter
|links| http://arxiv.org/abs/2402.14009v1 |
|updated| 2024-02-21 18:50:12 UTC |
|summary| We introduce the concept of geometry-informed neural networks GINNs whichencompass i learning under geometric constraints ii neural fields as asuitable representation and iii generating diverse solutions tounder-determined systems often encountered in geometric tasks. Notably theGINN formulation does not require training data and as such can be consideredgenerative modeling driven purely by constraints. We add an explicit diversityloss to mitigate mode collapse. We consider several constraints in particularthe connectedness of components which we convert to a differentiable lossthrough Morse theory. Experimentally we demonstrate the efficacy of the GINNlearning paradigm across a range of two and three-dimensional scenarios withincreasing levels of complexity. |


| Item |Content|
| --- |---|
|idx| 2402.14000v1 |
|title| Real-time 3D-aware Portrait Editing from a Single Image |
|authors| Qingyan BaiYinghao XuZifan ShiHao OuyangQiuyu WangCeyuan YangXuan WangGordon WetzsteinYujun ShenQifeng Chen
|links| http://arxiv.org/abs/2402.14000v1 |
|updated| 2024-02-21 18:36:26 UTC |
|summary| This work presents 3DPE a practical tool that can efficiently edit a faceimage following given prompts like reference images or text descriptions inthe 3D-aware manner. To this end a lightweight module is distilled from a 3Dportrait generator and a text-to-image model which provide prior knowledge offace geometry and open-vocabulary editing capability respectively. Such adesign brings two compelling advantages over existing approaches. First oursystem achieves real-time editing with a feedforward network i.e. 0.04s perimage over 100x faster than the second competitor. Second thanks to thepowerful priors our module could focus on the learning of editing-relatedvariations such that it manages to handle various types of editingsimultaneously in the training phase and further supports fast adaptation touser-specified novel types of editing during inference e.g. with 5minfine-tuning per case. The code the model and the interface will be madepublicly available to facilitate future research. |


| Item |Content|
| --- |---|
|idx| 2402.13955v1 |
|title| BEE-NET: A deep neural network to identify in-the-wild Bodily Expression of Emotions |
|authors| Mohammad Mahdi DehshibiDavid Masip
|links| http://arxiv.org/abs/2402.13955v1 |
|updated| 2024-02-21 17:35:51 UTC |
|summary| In this study we investigate how environmental factors specifically thescenes and objects involved can affect the expression of emotions through bodylanguage. To this end we introduce a novel multi-stream deep convolutionalneural network named BEE-NET. We also propose a new late fusion strategy thatincorporates meta-information on places and objects as prior knowledge in thelearning process. Our proposed probabilistic pooling model leverages thisinformation to generate a joint probability distribution of both available andanticipated non-available contextual information in latent space. Importantlyour fusion strategy is differentiable allowing for end-to-end training andcapturing of hidden associations among data points without requiring furtherpost-processing or regularisation. To evaluate our deep model we use the BodyLanguage Database BoLD which is currently the largest available database forthe Automatic Identification of the in-the-wild Bodily Expression of EmotionsAIBEE. Our experimental results demonstrate that our proposed approachsurpasses the current state-of-the-art in AIBEE by a margin of 2.07 achievingan Emotional Recognition Score of 66.33. |


| Item |Content|
| --- |---|
|idx| 2402.13936v1 |
|title| Distinctive Image Captioning: Leveraging Ground Truth Captions in CLIP Guided Reinforcement Learning |
|authors| Antoine ChaffinEwa KijakVincent Claveau
|links| http://arxiv.org/abs/2402.13936v1 |
|updated| 2024-02-21 17:05:06 UTC |
|summary| Training image captioning models using teacher forcing results in verygeneric samples whereas more distinctive captions can be very useful inretrieval applications or to produce alternative texts describing images foraccessibility. Reinforcement Learning RL allows to use cross-modal retrievalsimilarity score between the generated caption and the input image as reward toguide the training leading to more distinctive captions. Recent studies showthat pre-trained cross-modal retrieval models can be used to provide thisreward completely eliminating the need for reference captions. However weargue in this paper that Ground Truth GT captions can still be useful in thisRL framework. We propose a new image captioning model training strategy thatmakes use of GT captions in different ways. Firstly they can be used to traina simple MLP discriminator that serves as a regularization to prevent rewardhacking and ensures the fluency of generated captions resulting in a textualGAN setup extended for multimodal inputs. Secondly they can serve asadditional trajectories in the RL strategy resulting in a teacher forcing lossweighted by the similarity of the GT to the image. This objective acts as anadditional learning signal grounded to the distribution of the GT captions.Thirdly they can serve as strong baselines when added to the pool of captionsused to compute the proposed contrastive reward to reduce the variance ofgradient estimate. Experiments on MS-COCO demonstrate the interest of theproposed training strategy to produce highly distinctive captions whilemaintaining high writing quality. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.13999v1 |
|title| Asymptotics of Learning with Deep Structured (Random) Features |
|authors| Dominik SchröderDaniil DmitrievHugo CuiBruno Loureiro
|links| http://arxiv.org/abs/2402.13999v1 |
|updated| 2024-02-21 18:35:27 UTC |
|summary| For a large class of feature maps we provide a tight asymptoticcharacterisation of the test error associated with learning the readout layerin the high-dimensional limit where the input dimension hidden layer widthsand number of training samples are proportionally large. This characterizationis formulated in terms of the population covariance of the features. Our workis partially motivated by the problem of learning with Gaussian rainbow neuralnetworks namely deep non-linear fully-connected networks with random butstructured weights whose row-wise covariances are further allowed to depend onthe weights of previous layers. For such networks we also derive a closed-formformula for the feature covariance in terms of the weight matrices. We furtherfind that in some cases our results can capture feature maps learned by deepfinite-width neural networks trained under gradient descent. |


| Item |Content|
| --- |---|
|idx| 2402.13945v1 |
|title| Probabilistic Neural Networks (PNNs) for Modeling Aleatoric Uncertainty in Scientific Machine Learning |
|authors| Farhad Pourkamali-AnarakiJamal F. HusseiniScott E. Stapleton
|links| http://arxiv.org/abs/2402.13945v1 |
|updated| 2024-02-21 17:15:47 UTC |
|summary| This paper investigates the use of probabilistic neural networks PNNs tomodel aleatoric uncertainty which refers to the inherent variability in theinput-output relationships of a system often characterized by unequal varianceor heteroscedasticity. Unlike traditional neural networks that producedeterministic outputs PNNs generate probability distributions for the targetvariable allowing the determination of both predicted means and intervals inregression scenarios. Contributions of this paper include the development of aprobabilistic distance metric to optimize PNN architecture and the deploymentof PNNs in controlled data sets as well as a practical material science caseinvolving fiber-reinforced composites. The findings confirm that PNNseffectively model aleatoric uncertainty proving to be more appropriate thanthe commonly employed Gaussian process regression for this purpose.Specifically in a real-world scientific machine learning context PNNs yieldremarkably accurate output mean estimates with R-squared scores approaching0.97 and their predicted intervals exhibit a high correlation coefficient ofnearly 0.80 closely matching observed data intervals. Hence this researchcontributes to the ongoing exploration of leveraging the sophisticatedrepresentational capacity of neural networks to delineate complex input-outputrelationships in scientific problems. |


| Item |Content|
| --- |---|
|idx| 2402.13934v1 |
|title| Do Efficient Transformers Really Save Computation? |
|authors| Kai YangJan AckermannZhenyu HeGuhao FengBohang ZhangYunzhen FengQiwei YeDi HeLiwei Wang
|links| http://arxiv.org/abs/2402.13934v1 |
|updated| 2024-02-21 17:00:56 UTC |
|summary| As transformer-based language models are trained on increasingly largedatasets and with vast numbers of parameters finding more efficientalternatives to the standard Transformer has become very valuable. While manyefficient Transformers and Transformer alternatives have been proposed noneprovide theoretical guarantees that they are a suitable replacement for thestandard Transformer. This makes it challenging to identify when to use aspecific model and what directions to prioritize for further investigation. Inthis paper we aim to understand the capabilities and limitations of efficientTransformers specifically the Sparse Transformer and the Linear Transformer.We focus on their reasoning capability as exhibited by Chain-of-Thought CoTprompts and follow previous works to model them as Dynamic Programming DPproblems. Our results show that while these models are expressive enough tosolve general DP tasks contrary to expectations they require a model sizethat scales with the problem size. Nonetheless we identify a class of DPproblems for which these models can be more efficient than the standardTransformer. We confirm our theoretical results through experiments onrepresentative DP tasks adding to the understanding of efficient Transformerspractical strengths and weaknesses. |


| Item |Content|
| --- |---|
|idx| 2402.13903v1 |
|title| Dealing with unbounded gradients in stochastic saddle-point optimization |
|authors| Gergely NeuNneka Okolo
|links| http://arxiv.org/abs/2402.13903v1 |
|updated| 2024-02-21 16:13:49 UTC |
|summary| We study the performance of stochastic first-order methods for finding saddlepoints of convex-concave functions. A notorious challenge faced by such methodsis that the gradients can grow arbitrarily large during optimization which mayresult in instability and divergence. In this paper we propose a simple andeffective regularization technique that stabilizes the iterates and yieldsmeaningful performance guarantees even if the domain and the gradient noisescales linearly with the size of the iterates and is thus potentiallyunbounded. Besides providing a set of general results we also apply ouralgorithm to a specific problem in reinforcement learning where it leads toperformance guarantees for finding near-optimal policies in an average-rewardMDP without prior knowledge of the bias span. |


| Item |Content|
| --- |---|
|idx| 2402.13901v1 |
|title| Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate |
|authors| Yuchen LiangPeizhong JuYingbin LiangNess Shroff
|links| http://arxiv.org/abs/2402.13901v1 |
|updated| 2024-02-21 16:11:47 UTC |
|summary| The denoising diffusion model emerges recently as a powerful generativetechnique that converts noise into data. Theoretical convergence guarantee hasbeen mainly studied for continuous-time diffusion models and has been obtainedfor discrete-time diffusion models only for distributions with bounded supportin the literature. In this paper we establish the convergence guarantee forsubstantially larger classes of distributions under discrete-time diffusionmodels and further improve the convergence rate for distributions with boundedsupport. In particular we first establish the convergence rates for bothsmooth and general possibly non-smooth distributions having finite secondmoment. We then specialize our results to a number of interesting classes ofdistributions with explicit parameter dependencies including distributionswith Lipschitz scores Gaussian mixture distributions and distributions withbounded support. We further propose a novel accelerated sampler and show thatit improves the convergence rates of the corresponding regular sampler byorders of magnitude with respect to all system parameters. For distributionswith bounded support our result improves the dimensional dependence of theprevious convergence rate by orders of magnitude. Our study features a novelanalysis technique that constructs tilting factor representation of theconvergence error and exploits Tweedies formula for handling Taylor expansionpower terms. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.13992v1 |
|title| Meditating in Live Stream: An Autoethnographic and Interview Study to Investigate Motivations, Interactions and Challenges |
|authors| Jingjin LiJiajing GuoGilly Leshed
|links| http://dx.doi.org/10.1145/3637417 |
|updated| 2024-02-21 18:24:07 UTC |
|summary| Mindfulness practice has many mental and physical well-being benefits. Withthe increased popularity of live stream technologies and the impact ofCOVID-19 many people have turned to live stream tools to participate in onlinemeditation sessions. To better understand the practices challenges andopportunities in live-stream meditation we conducted a three-monthautoethnographic study during which two researchers participated inlive-stream meditation sessions as the audience. Then we conducted a follow-upsemi-structured interview study with 10 experienced live meditation teacherswho use different live-stream tools. We found that live meditation althoughhaving a weaker social presence than in-person meditation facilitatesattendees in establishing a practice routine and connecting with othermeditators. Teachers use live streams to deliver the meditation practice to theworld which also enhances their practice and brand building. We identified thechallenges of using live-stream tools for meditation from the perspectives ofboth audiences and teachers and provided design recommendations to betterutilize live meditation as a resource for mental wellbeing. |


| Item |Content|
| --- |---|
|idx| 2402.13939v1 |
|title| What is the focus of XAI in UI design? Prioritizing UI design principles for enhancing XAI user experience |
|authors| Dian LeiYao HeJianyou Zeng
|links| http://arxiv.org/abs/2402.13939v1 |
|updated| 2024-02-21 17:07:09 UTC |
|summary| With the widespread application of artificial intelligenceAI theexplainable AI XAI field has undergone a notable resurgence. In thisbackground the importance of user experience in XAI has become increasinglyprominent. Simultaneously the user interface UI serves as a crucial linkbetween XAI and users. However despite the existence of UI design principlesfor XAI there is a lack of prioritization based on their significance. Thiswill lead practitioners to have a vague understanding of different designprinciples making it difficult to allocate design space reasonably andemphasize design focal points. This paper aims to prioritize four designprinciples providing clear guidance for UI design in XAI. Initially weconducted a lightweight summary to derive five user experience standards fornon-expert users in XAI. Subsequently we developed four corresponding webpageprototypes for the four design principles. Nineteen participants theninteracted with these prototypes providing ratings based on five userexperience standards and We calculated the weights of the design principles.Our findings indicate that for non-expert users sensitivity is the optimalUI design principle weight  0.3296 followed by flexibility weight 0.3014. Finally we engage in further discussion and summarization of ourresearch results and present future works and limitations. |


| Item |Content|
| --- |---|
|idx| 2402.13771v1 |
|title| Mask-up: Investigating Biases in Face Re-identification for Masked Faces |
|authors| Siddharth D JaiswalAnkit Kr. VermaAnimesh Mukherjee
|links| http://arxiv.org/abs/2402.13771v1 |
|updated| 2024-02-21 12:48:45 UTC |
|summary| AI based Face Recognition Systems FRSs are now widely distributed anddeployed as MLaaS solutions all over the world moreso since the COVID-19pandemic for tasks ranging from validating individuals faces while buying SIMcards to surveillance of citizens. Extensive biases have been reported againstmarginalized groups in these systems and have led to highly discriminatoryoutcomes. The post-pandemic world has normalized wearing face masks but FRSshave not kept up with the changing times. As a result these systems aresusceptible to mask based face occlusion. In this study we audit fourcommercial and nine open-source FRSs for the task of face re-identificationbetween different varieties of masked and unmasked images across five benchmarkdatasets total 14722 images. These simulate a realisticvalidation/surveillance task as deployed in all major countries around theworld. Three of the commercial and five of the open-source FRSs are highlyinaccurate they further perpetuate biases against non-White individuals withthe lowest accuracy being 0. A survey for the same task with 85 humanparticipants also results in a low accuracy of 40. Thus a human-in-the-loopmoderation in the pipeline does not alleviate the concerns as has beenfrequently hypothesized in literature. Our large-scale study shows thatdevelopers lawmakers and users of such services need to rethink the designprinciples behind FRSs especially for the task of face re-identificationtaking cognizance of observed biases. |


| Item |Content|
| --- |---|
|idx| 2402.13724v1 |
|title| Bring Your Own Character: A Holistic Solution for Automatic Facial Animation Generation of Customized Characters |
|authors| Zechen BaiPeng ChenXiaolan PengLu LiuHui ChenMike Zheng ShouFeng Tian
|links| http://arxiv.org/abs/2402.13724v1 |
|updated| 2024-02-21 11:35:20 UTC |
|summary| Animating virtual characters has always been a fundamental research problemin virtual reality VR. Facial animations play a crucial role as theyeffectively convey emotions and attitudes of virtual humans. However creatingsuch facial animations can be challenging as current methods often involveutilization of expensive motion capture devices or significant investments oftime and effort from human animators in tuning animation parameters. In thispaper we propose a holistic solution to automatically animate virtual humanfaces. In our solution a deep learning model was first trained to retarget thefacial expression from input face images to virtual human faces by estimatingthe blendshape coefficients. This method offers the flexibility of generatinganimations with characters of different appearances and blendshape topologies.Second a practical toolkit was developed using Unity 3D making it compatiblewith the most popular VR applications. The toolkit accepts both image and videoas input to animate the target virtual human faces and enables users tomanipulate the animation results. Furthermore inspired by the spirit ofHuman-in-the-loop HITL we leveraged user feedback to further improve theperformance of the model and toolkit thereby increasing the customizationproperties to suit user preferences. The whole solution for which we will makethe code public has the potential to accelerate the generation of facialanimations for use in VR applications. |


| Item |Content|
| --- |---|
|idx| 2402.13688v1 |
|title| Exploring users' sense of safety in public using an Augmented Reality application |
|authors| Maurizio VergariTanja KojićNicole Stefanie BertgesFrancesco VonaSebastian MöllerJan-Niklas Voigt-Antons
|links| http://dx.doi.org/10.1109/QoMEX58391.2023.10178675 |
|updated| 2024-02-21 10:45:09 UTC |
|summary| Nowadays Augmented Reality AR is available on almost all smartphonescreating some exciting interaction opportunities but also challenges. Forexample already after the famous AR app Pokemon GO was released in July 2016numerous accidents related to the use of the app were reported by users. At thesame time the spread of AR can be noticed in the tourism industry enablingtourists to explore their surroundings in new ways but also exposing them tosafety issues. This preliminary study explores users sense of safety whenmanipulating the amount and UI elements visualization parameters of Point ofInterest POI markers in a developed AR application. The results show that theamount of POI markers that are displayed is significant for participants senseof safety. The influence of manipulating UI elements in terms of transparencycolor and size cannot be proven. Nevertheless most tested people stated thatmanipulating transparency and size somehow influences their sense of safety soa closer look at them should be taken in future studies. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.13219v1 |
|title| Analyzing Operator States and the Impact of AI-Enhanced Decision Support in Control Rooms: A Human-in-the-Loop Specialized Reinforcement Learning Framework for Intervention Strategies |
|authors| Ammar N. AbbasChidera W. AmazuJoseph MietkiewiczHouda BriwaAndres Alonzo PerezGabriele BaldissoneMicaela DemichelaGeorgios G. ChasparisJohn D. KelleherMaria Chiara Leva
|links| http://arxiv.org/abs/2402.13219v1 |
|updated| 2024-02-20 18:31:27 UTC |
|summary| In complex industrial and chemical process control rooms effectivedecision-making is crucial for safety and efficiency. The experiments in thispaper evaluate the impact and applications of an AI-based decision supportsystem integrated into an improved human-machine interface using dynamicinfluence diagrams a hidden Markov model and deep reinforcement learning. Theenhanced support system aims to reduce operator workload improve situationalawareness and provide different intervention strategies to the operatoradapted to the current state of both the system and human performance. Such asystem can be particularly useful in cases of information overload when manyalarms and inputs are presented all within the same time window or for junioroperators during training. A comprehensive cross-data analysis was conductedinvolving 47 participants and a diverse range of data sources such assmartwatch metrics eye-tracking data process logs and responses fromquestionnaires. The results indicate interesting insights regarding theeffectiveness of the approach in aiding decision-making decreasing perceivedworkload and increasing situational awareness for the scenarios considered.Additionally the results provide valuable insights to compare differencesbetween styles of information gathering when using the system by individualparticipants. These findings are particularly relevant when predicting theoverall performance of the individual participant and their capacity tosuccessfully handle a plant upset and the alarms connected to it using processand human-machine interaction logs in real-time. These predictions enable thedevelopment of more effective intervention strategies. |


| Item |Content|
| --- |---|
|idx| 2402.13292v1 |
|title| A Conflict-Aware Optimal Goal Assignment Algorithm for Multi-Robot Systems |
|authors| AakashIndranil Saha
|links| http://arxiv.org/abs/2402.13292v1 |
|updated| 2024-02-19 19:04:19 UTC |
|summary| The fundamental goal assignment problem for a multi-robot application aims toassign a unique goal to each robot while ensuring collision-free pathsminimizing the total movement cost. A plausible algorithmic solution to thisNP-hard problem involves an iterative process that integrates a task planner tocompute the goal assignment while ignoring the collision possibilities amongthe robots and a multi-agent path-finding algorithm to find the collision-freetrajectories for a given assignment. This procedure involves a method forcomputing the next best assignment given the current best assignment. A naiveway of computing the next best assignment as done in the state-of-the-artsolutions becomes a roadblock to achieving scalability in solving the overallproblem. To obviate this bottleneck we propose an efficient conflict-guidedmethod to compute the next best assignment. Additionally we introduce two moreoptimizations to the algorithm -- first for avoiding the unconstrained pathcomputations between robot-goal pairs wherever possible and the second toprevent duplicate constrained path computations for multiple robot-goal pairs.We extensively evaluate our algorithm for up to a hundred robots on severalbenchmark workspaces. The results demonstrate that the proposed algorithmachieves nearly an order of magnitude speedup over the state-of-the-artalgorithm showcasing its efficacy in real-world scenarios. |


| Item |Content|
| --- |---|
|idx| 2402.12327v1 |
|title| Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents |
|authors| Zengqing WuShuyuan ZhengQianying LiuXu HanBrian Inhyuk KwonMakoto OnizukaShaojie TangRun PengChuan Xiao
|links| http://arxiv.org/abs/2402.12327v1 |
|updated| 2024-02-19 18:00:53 UTC |
|summary| Recent advancements have shown that agents powered by large language modelsLLMs possess capabilities to simulate human behaviors and societal dynamics.However the potential for LLM agents to spontaneously establish collaborativerelationships in the absence of explicit instructions has not been studied. Toaddress this gap we conduct three case studies revealing that LLM agents arecapable of spontaneously forming collaborations even within competitivesettings. This finding not only demonstrates the capacity of LLM agents tomimic competition and cooperation in human societies but also validates apromising vision of computational social science. Specifically it suggeststhat LLM agents could be utilized to model human social interactions includingthose with spontaneous collaborations thus offering insights into socialphenomena. The source codes for this study are available athttps://github.com/wuzengqing001225/SABM_ShallWeTalk . |


| Item |Content|
| --- |---|
|idx| 2402.12326v1 |
|title| LLM Agents for Psychology: A Study on Gamified Assessments |
|authors| Qisen YangZekun WangHonghui ChenShenzhi WangYifan PuXin GaoWenhao HuangShiji SongGao Huang
|links| http://arxiv.org/abs/2402.12326v1 |
|updated| 2024-02-19 18:00:30 UTC |
|summary| Psychological measurement is essential for mental health self-understandingand personal development. Traditional methods such as self-report scales andpsychologist interviews often face challenges with engagement andaccessibility. While game-based and LLM-based tools have been explored toimprove user interest and automate assessment they struggle to balanceengagement with generalizability. In this work we propose PsychoGATPsychological Game AgenTs to achieve a generic gamification of psychologicalassessment. The main insight is that powerful LLMs can function both as adeptpsychologists and innovative game designers. By incorporating LLM agents intodesignated roles and carefully managing their interactions PsychoGAT cantransform any standardized scales into personalized and engaging interactivefiction games. To validate the proposed method we conduct psychometricevaluations to assess its effectiveness and employ human evaluators to examinethe generated content across various psychological constructs includingdepression cognitive distortions and personality traits. Results demonstratethat PsychoGAT serves as an effective assessment tool achieving statisticallysignificant excellence in psychometric metrics such as reliability convergentvalidity and discriminant validity. Moreover human evaluations confirmPsychoGATs enhancements in content coherence interactivity interestimmersion and satisfaction. |


| Item |Content|
| --- |---|
|idx| 2402.12086v1 |
|title| Navigating simplicity and complexity of social-ecological systems through a dialog between dynamical systems and agent-based models |
|authors| Sonja RadosavljevicUdita SangaMaja Schlüter
|links| http://arxiv.org/abs/2402.12086v1 |
|updated| 2024-02-19 12:07:27 UTC |
|summary| Social-ecological systems SES research aims to understand the nature ofsocial-ecological phenomena to find effective ways to foster or manageconditions under which desirable phenomena such as sustainable resource useoccur or to change conditions or reduce the negative consequences ofundesirable phenomena such as poverty traps. Challenges such as these areoften addressed using dynamical systems models DSM or agent-based modelsABM. Both modeling approaches have strengths and weaknesses. DSM are praisedfor their analytical tractability and efficient exploration of asymptoticdynamics and bifurcation which are enabled by reduced number and heterogeneityof system components. ABM allows representing heterogeneity agency learningand interactions of diverse agents within SES but this also comes at a pricesuch as inefficiency to explore asymptotic dynamics or bifurcations. In thispaper we combine DSM and ABM to leverage strengths of each modeling techniqueand gain deeper insights into dynamics of a system. We start with an ABM andresearch questions that the ABM was not able to answer. Using results of theABM analysis as inputs for DSM we create a DSM. Stability and bifurcationanalysis of the DSM gives partial answers to the research questions and directattention to where additional details are needed. This informs further ABManalysis prevents burdening the ABM with less important details and revealsnew insights about system dynamics. The iterative process and dialogue betweenthe ABM and DSM leads to more complete answers to research questions andsurpasses insights provided by each of the models separately. We illustrate theprocedure with the example of the emergence of poverty traps in an agriculturalsystem with endogenously driven innovation. |


