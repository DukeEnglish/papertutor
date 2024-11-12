# cs.CL 

| Item |Content|
| --- |---|
|idx| 2411.05787v1 |
|title| Recycled Attention: Efficient inference for long-context language models |
|authors| Fangyuan XuTanya GoyalEunsol Choi
|links| http://arxiv.org/abs/2411.05787v1 |
|updated| 2024-11-08 18:57:07 UTC |
|summary| Generating long sequences of tokens given a long-context input imposes aheavy computational burden for large language models LLMs. One of thecomputational bottleneck comes from computing attention over a long sequence ofinput at each generation step. In this paper we propose Recycled Attention aninference-time method which alternates between full context attention andattention over a subset of input tokens. When performing partial attention werecycle the attention pattern of a previous token that has performed fullattention and attend only to the top K most attended tokens reducing the costof data movement and attention computation. Compared to previously proposedinference-time acceleration method which attends only to local context ortokens with high accumulative attention scores our approach flexibly choosestokens that are relevant to the current decoding step. We evaluate our methodson RULER a suite of tasks designed to comprehensively evaluate long-contextabilities and long-context language modeling tasks. Applying our method tooff-the-shelf LLMs achieves comparable speedup to baselines which only considerlocal context while improving the performance by 2x. We further explore twoideas to improve performance-efficiency trade-offs: 1 dynamically decide whento perform recycled or full attention step based on the query similarities and2 continued pre-training the model with Recycled Attention. |


| Item |Content|
| --- |---|
|idx| 2411.05783v1 |
|title| ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles |
|authors| Kayo YinChinmay SinghFyodor O. MinakovVanessa MilanHal Daumé IIICyril ZhangAlex X. LuDanielle Bragg
|links| http://arxiv.org/abs/2411.05783v1 |
|updated| 2024-11-08 18:50:37 UTC |
|summary| Deaf and hard-of-hearing DHH students face significant barriers inaccessing science technology engineering and mathematics STEM educationnotably due to the scarcity of STEM resources in signed languages. To helpaddress this we introduce ASL STEM Wiki: a parallel corpus of 254 Wikipediaarticles on STEM topics in English interpreted into over 300 hours of AmericanSign Language ASL. ASL STEM Wiki is the first continuous signing datasetfocused on STEM facilitating the development of AI resources for STEMeducation in ASL. We identify several use cases of ASL STEM Wiki withhuman-centered applications. For example because this dataset highlights thefrequent use of fingerspelling for technical concepts which inhibits DHHstudents ability to learn we develop models to identify fingerspelled words-- which can later be used to query for appropriate ASL signs to suggest tointerpreters. |


| Item |Content|
| --- |---|
|idx| 2411.05781v1 |
|title| Using Language Models to Disambiguate Lexical Choices in Translation |
|authors| Josh BaruaSanjay SubramanianKayo YinAlane Suhr
|links| http://arxiv.org/abs/2411.05781v1 |
|updated| 2024-11-08 18:48:57 UTC |
|summary| In translation a concept represented by a single word in a source languagecan have multiple variations in a target language. The task of lexicalselection requires using context to identify which variation is mostappropriate for a source text. We work with native speakers of nine languagesto create DTAiLS a dataset of 1377 sentence pairs that exhibit cross-lingualconcept variation when translating from English. We evaluate recent LLMs andneural machine translation systems on DTAiLS with the best-performing modelGPT-4 achieving from 67 to 85 accuracy across languages. Finally we uselanguage models to generate English rules describing target-language conceptvariations. Providing weaker models with high-quality lexical rules improvesaccuracy substantially in some cases reaching or outperforming GPT-4. |


| Item |Content|
| --- |---|
|idx| 2411.05778v1 |
|title| LLMs as Method Actors: A Model for Prompt Engineering and Architecture |
|authors| Colin Doyle
|links| http://arxiv.org/abs/2411.05778v1 |
|updated| 2024-11-08 18:45:06 UTC |
|summary| We introduce Method Actors as a mental model for guiding LLM promptengineering and prompt architecture. Under this mental model LLMs should bethought of as actors prompts as scripts and cues and LLM responses asperformances. We apply this mental model to the task of improving LLMperformance at playing Connections a New York Times word puzzle game thatprior research identified as a challenging benchmark for evaluating LLMreasoning. Our experiments with GPT-4o show that a Method Actors approach cansignificantly improve LLM performance over both a vanilla and Chain ofThoughts approach. A vanilla approach solves 27 of Connections puzzles in ourdataset and a Chain of Thoughts approach solves 41 of puzzles whereas ourstrongest Method Actor approach solves 86 of puzzles. We also test OpenAIsnewest model designed specifically for complex reasoning tasks o1-preview.When asked to solve a puzzle all at once o1-preview solves 79 of Connectionspuzzles in our dataset and when allowed to build puzzle solutions one guess ata time over multiple API calls o1-preview solves 100 of the puzzles.Incorporating a Method Actor prompt architecture increases the percentage ofpuzzles that o1-preview solves perfectly from 76 to 87. |


| Item |Content|
| --- |---|
|idx| 2411.05777v1 |
|title| Quantitative Assessment of Intersectional Empathetic Bias and Understanding |
|authors| Vojtech FormanekOndrej Sotolar
|links| http://arxiv.org/abs/2411.05777v1 |
|updated| 2024-11-08 18:43:15 UTC |
|summary| A growing amount of literature critiques the current operationalizations ofempathy based on loose definitions of the construct. Such definitionsnegatively affect dataset quality model robustness and evaluationreliability. We propose an empathy evaluation framework that operationalizesempathy close to its psychological origins. The framework measures the variancein responses of LLMs to prompts using existing metrics for empathy andemotional valence. The variance is introduced through the controlled generationof the prompts by varying social biases affecting context understanding thusimpacting empathetic understanding. The control over generation ensures hightheoretical validity of the constructs in the prompt dataset. Also it makeshigh-quality translation especially into languages that currently havelittle-to-no way of evaluating empathy or bias such as the Slavonic familymore manageable. Using chosen LLMs and various prompt types we demonstrate theempathy evaluation with the framework including multiple-choice answers andfree generation. The variance in our initial evaluation sample is small and wewere unable to measure convincing differences between the empatheticunderstanding in contexts given by different social groups. However theresults are promising because the models showed significant alterations theirreasoning chains needed to capture the relatively subtle changes in theprompts. This provides the basis for future research into the construction ofthe evaluation sample and statistical methods for measuring the results. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2411.05783v1 |
|title| ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles |
|authors| Kayo YinChinmay SinghFyodor O. MinakovVanessa MilanHal Daumé IIICyril ZhangAlex X. LuDanielle Bragg
|links| http://arxiv.org/abs/2411.05783v1 |
|updated| 2024-11-08 18:50:37 UTC |
|summary| Deaf and hard-of-hearing DHH students face significant barriers inaccessing science technology engineering and mathematics STEM educationnotably due to the scarcity of STEM resources in signed languages. To helpaddress this we introduce ASL STEM Wiki: a parallel corpus of 254 Wikipediaarticles on STEM topics in English interpreted into over 300 hours of AmericanSign Language ASL. ASL STEM Wiki is the first continuous signing datasetfocused on STEM facilitating the development of AI resources for STEMeducation in ASL. We identify several use cases of ASL STEM Wiki withhuman-centered applications. For example because this dataset highlights thefrequent use of fingerspelling for technical concepts which inhibits DHHstudents ability to learn we develop models to identify fingerspelled words-- which can later be used to query for appropriate ASL signs to suggest tointerpreters. |


| Item |Content|
| --- |---|
|idx| 2411.05781v1 |
|title| Using Language Models to Disambiguate Lexical Choices in Translation |
|authors| Josh BaruaSanjay SubramanianKayo YinAlane Suhr
|links| http://arxiv.org/abs/2411.05781v1 |
|updated| 2024-11-08 18:48:57 UTC |
|summary| In translation a concept represented by a single word in a source languagecan have multiple variations in a target language. The task of lexicalselection requires using context to identify which variation is mostappropriate for a source text. We work with native speakers of nine languagesto create DTAiLS a dataset of 1377 sentence pairs that exhibit cross-lingualconcept variation when translating from English. We evaluate recent LLMs andneural machine translation systems on DTAiLS with the best-performing modelGPT-4 achieving from 67 to 85 accuracy across languages. Finally we uselanguage models to generate English rules describing target-language conceptvariations. Providing weaker models with high-quality lexical rules improvesaccuracy substantially in some cases reaching or outperforming GPT-4. |


| Item |Content|
| --- |---|
|idx| 2411.05780v1 |
|title| GazeSearch: Radiology Findings Search Benchmark |
|authors| Trong Thang PhamTien-Phat NguyenYuki IkebeAkash AwasthiZhigang DengCarol C. WuHien NguyenNgan Le
|links| http://arxiv.org/abs/2411.05780v1 |
|updated| 2024-11-08 18:47:08 UTC |
|summary| Medical eye-tracking data is an important information source forunderstanding how radiologists visually interpret medical images. Thisinformation not only improves the accuracy of deep learning models for X-rayanalysis but also their interpretability enhancing transparency indecision-making. However the current eye-tracking data is dispersedunprocessed and ambiguous making it difficult to derive meaningful insights.Therefore there is a need to create a new dataset with more focus andpurposeful eyetracking data improving its utility for diagnostic applications.In this work we propose a refinement method inspired by the target-presentvisual search challenge: there is a specific finding and fixations are guidedto locate it. After refining the existing eye-tracking datasets we transformthem into a curated visual search dataset called GazeSearch specifically forradiology findings where each fixation sequence is purposefully aligned to thetask of locating a particular finding. Subsequently we introduce a scan pathprediction baseline called ChestSearch specifically tailored to GazeSearch.Finally we employ the newly introduced GazeSearch as a benchmark to evaluatethe performance of current state-of-the-art methods offering a comprehensiveassessment for visual search in the medical imaging domain. |


| Item |Content|
| --- |---|
|idx| 2411.05778v1 |
|title| LLMs as Method Actors: A Model for Prompt Engineering and Architecture |
|authors| Colin Doyle
|links| http://arxiv.org/abs/2411.05778v1 |
|updated| 2024-11-08 18:45:06 UTC |
|summary| We introduce Method Actors as a mental model for guiding LLM promptengineering and prompt architecture. Under this mental model LLMs should bethought of as actors prompts as scripts and cues and LLM responses asperformances. We apply this mental model to the task of improving LLMperformance at playing Connections a New York Times word puzzle game thatprior research identified as a challenging benchmark for evaluating LLMreasoning. Our experiments with GPT-4o show that a Method Actors approach cansignificantly improve LLM performance over both a vanilla and Chain ofThoughts approach. A vanilla approach solves 27 of Connections puzzles in ourdataset and a Chain of Thoughts approach solves 41 of puzzles whereas ourstrongest Method Actor approach solves 86 of puzzles. We also test OpenAIsnewest model designed specifically for complex reasoning tasks o1-preview.When asked to solve a puzzle all at once o1-preview solves 79 of Connectionspuzzles in our dataset and when allowed to build puzzle solutions one guess ata time over multiple API calls o1-preview solves 100 of the puzzles.Incorporating a Method Actor prompt architecture increases the percentage ofpuzzles that o1-preview solves perfectly from 76 to 87. |


| Item |Content|
| --- |---|
|idx| 2411.05777v1 |
|title| Quantitative Assessment of Intersectional Empathetic Bias and Understanding |
|authors| Vojtech FormanekOndrej Sotolar
|links| http://arxiv.org/abs/2411.05777v1 |
|updated| 2024-11-08 18:43:15 UTC |
|summary| A growing amount of literature critiques the current operationalizations ofempathy based on loose definitions of the construct. Such definitionsnegatively affect dataset quality model robustness and evaluationreliability. We propose an empathy evaluation framework that operationalizesempathy close to its psychological origins. The framework measures the variancein responses of LLMs to prompts using existing metrics for empathy andemotional valence. The variance is introduced through the controlled generationof the prompts by varying social biases affecting context understanding thusimpacting empathetic understanding. The control over generation ensures hightheoretical validity of the constructs in the prompt dataset. Also it makeshigh-quality translation especially into languages that currently havelittle-to-no way of evaluating empathy or bias such as the Slavonic familymore manageable. Using chosen LLMs and various prompt types we demonstrate theempathy evaluation with the framework including multiple-choice answers andfree generation. The variance in our initial evaluation sample is small and wewere unable to measure convincing differences between the empatheticunderstanding in contexts given by different social groups. However theresults are promising because the models showed significant alterations theirreasoning chains needed to capture the relatively subtle changes in theprompts. This provides the basis for future research into the construction ofthe evaluation sample and statistical methods for measuring the results. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2411.05779v1 |
|title| Curriculum Learning for Few-Shot Domain Adaptation in CT-based Airway Tree Segmentation |
|authors| Maxime JacovellaAli KeshavarziElsa Angelini
|links| http://arxiv.org/abs/2411.05779v1 |
|updated| 2024-11-08 18:46:40 UTC |
|summary| Despite advances with deep learning DL automated airway segmentation fromchest CT scans continues to face challenges in segmentation quality andgeneralization across cohorts. To address these we propose integratingCurriculum Learning CL into airway segmentation networks distributing thetraining set into batches according to ad-hoc complexity scores derived from CTscans and corresponding ground-truth tree features. We specifically investigatefew-shot domain adaptation targeting scenarios where manual annotation of afull fine-tuning dataset is prohibitively expensive. Results are reported ontwo large open-cohorts ATM22 and AIIB23 with high performance using CL forfull training Source domain and few-shot fine-tuning Target domain butwith also some insights on potential detrimental effects if using a classicBootstrapping scoring function or if not using proper scan sequencing. |


| Item |Content|
| --- |---|
|idx| 2411.05771v1 |
|title| Sketched Equivariant Imaging Regularization and Deep Internal Learning for Inverse Problems |
|authors| Guixian XuJinglai LiJunqi Tang
|links| http://arxiv.org/abs/2411.05771v1 |
|updated| 2024-11-08 18:33:03 UTC |
|summary| Equivariant Imaging EI regularization has become the de-facto technique forunsupervised training of deep imaging networks without any need ofground-truth data. Observing that the EI-based unsupervised training paradigmcurrently has significant computational redundancy leading to inefficiency inhigh-dimensional applications we propose a sketched EI regularization whichleverages the randomized sketching techniques for acceleration. We then extendour sketched EI regularization to develop an accelerated deep internal learningframework -- Sketched Equivariant Deep Image Prior Sk.EI-DIP which can beefficiently applied for single-image and task-adapted reconstruction. Ournumerical study on X-ray CT image reconstruction tasks demonstrate that ourapproach can achieve order-of-magnitude computational acceleration overstandard EI-based counterpart in single-input setting and network adaptationat test time. |


| Item |Content|
| --- |---|
|idx| 2411.05764v1 |
|title| FinDVer: Explainable Claim Verification over Long and Hybrid-Content Financial Documents |
|authors| Yilun ZhaoYitao LongYuru JiangChengye WangWeiyuan ChenHongjun LiuYiming ZhangXiangru TangChen ZhaoArman Cohan
|links| http://arxiv.org/abs/2411.05764v1 |
|updated| 2024-11-08 18:26:17 UTC |
|summary| We introduce FinDVer a comprehensive benchmark specifically designed toevaluate the explainable claim verification capabilities of LLMs in the contextof understanding and analyzing long hybrid-content financial documents.FinDVer contains 2400 expert-annotated examples divided into three subsets:information extraction numerical reasoning and knowledge-intensive reasoningeach addressing common scenarios encountered in real-world financial contexts.We assess a broad spectrum of LLMs under long-context and RAG settings. Ourresults show that even the current best-performing system GPT-4o still lagsbehind human experts. We further provide in-depth analysis on long-context andRAG setting Chain-of-Thought reasoning and model reasoning errors offeringinsights to drive future advancements. We believe that FinDVer can serve as avaluable benchmark for evaluating LLMs in claim verification over complexexpert-domain documents. |


| Item |Content|
| --- |---|
|idx| 2411.05757v1 |
|title| Tract-RLFormer: A Tract-Specific RL policy based Decoder-only Transformer Network |
|authors| Ankita JoshiAshutosh SharmaAnoushkrit GoelRanjeet Ranjan JhaChirag AhujaArnav BhavsarAditya Nigam
|links| http://arxiv.org/abs/2411.05757v1 |
|updated| 2024-11-08 18:18:18 UTC |
|summary| Fiber tractography is a cornerstone of neuroimaging enabling the detailedmapping of the brains white matter pathways through diffusion MRI. This iscrucial for understanding brain connectivity and function making it a valuabletool in neurological applications. Despite its importance tractography faceschallenges due to its complexity and susceptibility to false positivesmisrepresenting vital pathways. To address these issues recent strategies haveshifted towards deep learning utilizing supervised learning which depends onprecise ground truth or reinforcement learning which operates without it. Inthis work we propose Tract-RLFormer a network utilizing both supervised andreinforcement learning in a two-stage policy refinement process that markedlyimproves the accuracy and generalizability across various data-sets. Byemploying a tract-specific approach our network directly delineates the tractsof interest bypassing the traditional segmentation process. Through rigorousvalidation on datasets such as TractoInferno HCP and ISMRM-2015 ourmethodology demonstrates a leap forward in tractography showcasing its abilityto accurately map the brains white matter tracts. |


| Item |Content|
| --- |---|
|idx| 2411.05752v1 |
|title| FisherMask: Enhancing Neural Network Labeling Efficiency in Image Classification Using Fisher Information |
|authors| Shreen GulMohamed ElmahallawySanjay MadriaArdhendu Tripathy
|links| http://arxiv.org/abs/2411.05752v1 |
|updated| 2024-11-08 18:10:46 UTC |
|summary| Deep learning DL models are popular across various domains due to theirremarkable performance and efficiency. However their effectiveness reliesheavily on large amounts of labeled data which are often time-consuming andlabor-intensive to generate manually. To overcome this challenge it isessential to develop strategies that reduce reliance on extensive labeled datawhile preserving model performance. In this paper we propose FisherMask aFisher information-based active learning AL approach that identifies keynetwork parameters by masking them based on their Fisher information values.FisherMask enhances batch AL by using Fisher information to select the mostcritical parameters allowing the identification of the most impactful samplesduring AL training. Moreover Fisher information possesses favorablestatistical properties offering valuable insights into model behavior andproviding a better understanding of the performance characteristics within theAL pipeline. Our extensive experiments demonstrate that FisherMasksignificantly outperforms state-of-the-art methods on diverse datasetsincluding CIFAR-10 and FashionMNIST especially under imbalanced settings.These improvements lead to substantial gains in labeling efficiency. Henceserving as an effective tool to measure the sensitivity of model parameters todata samples. Our code is available onurlhttps://github.com/sgchr273/FisherMask. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2411.05783v1 |
|title| ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles |
|authors| Kayo YinChinmay SinghFyodor O. MinakovVanessa MilanHal Daumé IIICyril ZhangAlex X. LuDanielle Bragg
|links| http://arxiv.org/abs/2411.05783v1 |
|updated| 2024-11-08 18:50:37 UTC |
|summary| Deaf and hard-of-hearing DHH students face significant barriers inaccessing science technology engineering and mathematics STEM educationnotably due to the scarcity of STEM resources in signed languages. To helpaddress this we introduce ASL STEM Wiki: a parallel corpus of 254 Wikipediaarticles on STEM topics in English interpreted into over 300 hours of AmericanSign Language ASL. ASL STEM Wiki is the first continuous signing datasetfocused on STEM facilitating the development of AI resources for STEMeducation in ASL. We identify several use cases of ASL STEM Wiki withhuman-centered applications. For example because this dataset highlights thefrequent use of fingerspelling for technical concepts which inhibits DHHstudents ability to learn we develop models to identify fingerspelled words-- which can later be used to query for appropriate ASL signs to suggest tointerpreters. |


| Item |Content|
| --- |---|
|idx| 2411.05780v1 |
|title| GazeSearch: Radiology Findings Search Benchmark |
|authors| Trong Thang PhamTien-Phat NguyenYuki IkebeAkash AwasthiZhigang DengCarol C. WuHien NguyenNgan Le
|links| http://arxiv.org/abs/2411.05780v1 |
|updated| 2024-11-08 18:47:08 UTC |
|summary| Medical eye-tracking data is an important information source forunderstanding how radiologists visually interpret medical images. Thisinformation not only improves the accuracy of deep learning models for X-rayanalysis but also their interpretability enhancing transparency indecision-making. However the current eye-tracking data is dispersedunprocessed and ambiguous making it difficult to derive meaningful insights.Therefore there is a need to create a new dataset with more focus andpurposeful eyetracking data improving its utility for diagnostic applications.In this work we propose a refinement method inspired by the target-presentvisual search challenge: there is a specific finding and fixations are guidedto locate it. After refining the existing eye-tracking datasets we transformthem into a curated visual search dataset called GazeSearch specifically forradiology findings where each fixation sequence is purposefully aligned to thetask of locating a particular finding. Subsequently we introduce a scan pathprediction baseline called ChestSearch specifically tailored to GazeSearch.Finally we employ the newly introduced GazeSearch as a benchmark to evaluatethe performance of current state-of-the-art methods offering a comprehensiveassessment for visual search in the medical imaging domain. |


| Item |Content|
| --- |---|
|idx| 2411.05779v1 |
|title| Curriculum Learning for Few-Shot Domain Adaptation in CT-based Airway Tree Segmentation |
|authors| Maxime JacovellaAli KeshavarziElsa Angelini
|links| http://arxiv.org/abs/2411.05779v1 |
|updated| 2024-11-08 18:46:40 UTC |
|summary| Despite advances with deep learning DL automated airway segmentation fromchest CT scans continues to face challenges in segmentation quality andgeneralization across cohorts. To address these we propose integratingCurriculum Learning CL into airway segmentation networks distributing thetraining set into batches according to ad-hoc complexity scores derived from CTscans and corresponding ground-truth tree features. We specifically investigatefew-shot domain adaptation targeting scenarios where manual annotation of afull fine-tuning dataset is prohibitively expensive. Results are reported ontwo large open-cohorts ATM22 and AIIB23 with high performance using CL forfull training Source domain and few-shot fine-tuning Target domain butwith also some insights on potential detrimental effects if using a classicBootstrapping scoring function or if not using proper scan sequencing. |


| Item |Content|
| --- |---|
|idx| 2411.05771v1 |
|title| Sketched Equivariant Imaging Regularization and Deep Internal Learning for Inverse Problems |
|authors| Guixian XuJinglai LiJunqi Tang
|links| http://arxiv.org/abs/2411.05771v1 |
|updated| 2024-11-08 18:33:03 UTC |
|summary| Equivariant Imaging EI regularization has become the de-facto technique forunsupervised training of deep imaging networks without any need ofground-truth data. Observing that the EI-based unsupervised training paradigmcurrently has significant computational redundancy leading to inefficiency inhigh-dimensional applications we propose a sketched EI regularization whichleverages the randomized sketching techniques for acceleration. We then extendour sketched EI regularization to develop an accelerated deep internal learningframework -- Sketched Equivariant Deep Image Prior Sk.EI-DIP which can beefficiently applied for single-image and task-adapted reconstruction. Ournumerical study on X-ray CT image reconstruction tasks demonstrate that ourapproach can achieve order-of-magnitude computational acceleration overstandard EI-based counterpart in single-input setting and network adaptationat test time. |


| Item |Content|
| --- |---|
|idx| 2411.05755v1 |
|title| End-to-End Navigation with Vision Language Models: Transforming Spatial Reasoning into Question-Answering |
|authors| Dylan GoettingHimanshu Gaurav SinghAntonio Loquercio
|links| http://arxiv.org/abs/2411.05755v1 |
|updated| 2024-11-08 18:16:58 UTC |
|summary| We present VLMnav an embodied framework to transform a Vision-Language ModelVLM into an end-to-end navigation policy. In contrast to prior work we donot rely on a separation between perception planning and control instead weuse a VLM to directly select actions in one step. Surprisingly we find that aVLM can be used as an end-to-end policy zero-shot i.e. without anyfine-tuning or exposure to navigation data. This makes our approach open-endedand generalizable to any downstream navigation task. We run an extensive studyto evaluate the performance of our approach in comparison to baseline promptingmethods. In addition we perform a design analysis to understand the mostimpactful design decisions. Visual examples and code for our project can befound at https://jirl-upenn.github.io/VLMnav/ |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2411.05750v1 |
|title| On Differentially Private String Distances |
|authors| Jerry Yao-Chieh HuErzhi LiuHan LiuZhao SongLichen Zhang
|links| http://arxiv.org/abs/2411.05750v1 |
|updated| 2024-11-08 18:10:07 UTC |
|summary| Given a database of bit strings A_1ldotsA_min 01n a fundamentaldata structure task is to estimate the distances between a given query Bin01n with all the strings in the database. In addition one might furtherwant to ensure the integrity of the database by releasing these distancestatistics in a secure manner. In this work we propose differentially privateDP data structures for this type of tasks with a focus on Hamming and editdistance. On top of the strong privacy guarantees our data structures are alsotime- and space-efficient. In particular our data structure is epsilon-DPagainst any sequence of queries of arbitrary length and for any query B suchthat the maximum distance to any string in the database is at most k weoutput m distance estimates. Moreover  - For Hamming distance our data structure answers any query in widetildeOmkn time and each estimate deviates from the true distance by at mostwidetilde Ok/eepsilon/log k  - For edit distance our data structure answers any query in widetildeOmk2n time and each estimate deviates from the true distance by at mostwidetilde Ok/eepsilon/log k log n.  For moderate k both data structures support sublinear query operations. Weobtain these results via a novel adaptation of the randomized responsetechnique as a bit flipping procedure applied to the sketched strings. |


| Item |Content|
| --- |---|
|idx| 2411.05735v1 |
|title| Aioli: A Unified Optimization Framework for Language Model Data Mixing |
|authors| Mayee F. ChenMichael Y. HuNicholas LourieKyunghyun ChoChristopher Ré
|links| http://arxiv.org/abs/2411.05735v1 |
|updated| 2024-11-08 17:50:24 UTC |
|summary| Language model performance depends on identifying the optimal mixture of datagroups to train on e.g. law code math. Prior work has proposed a diverseset of methods to efficiently learn mixture proportions ranging from fittingregression models over training runs to dynamically updating proportionsthroughout training. Surprisingly we find that no existing method consistentlyoutperforms a simple stratified sampling baseline in terms of average testperplexity per group. In this paper we study the cause of this inconsistencyby unifying existing methods into a standard optimization framework. We showthat all methods set proportions to minimize total loss subject to amethod-specific mixing law -- an assumption on how loss is a function ofmixture proportions. We find that existing parameterizations of mixing laws canexpress the true loss-proportion relationship empirically but the methodsthemselves often set the mixing law parameters inaccurately resulting in poorand inconsistent performance. Finally we leverage the insights from ourframework to derive a new online method named Aioli which directly estimatesthe mixing law parameters throughout training and uses them to dynamicallyadjust proportions. Empirically Aioli outperforms stratified sampling on 6 outof 6 datasets by an average of 0.28 test perplexity points whereas existingmethods fail to consistently beat stratified sampling doing up to 6.9 pointsworse. Moreover in a practical setting where proportions are learned onshorter runs due to computational constraints Aioli can dynamically adjustthese proportions over the full training run consistently improvingperformance over existing methods by up to 12.01 test perplexity points. |


| Item |Content|
| --- |---|
|idx| 2411.05729v1 |
|title| Graph-Dictionary Signal Model for Sparse Representations of Multivariate Data |
|authors| William CappellettiPascal Frossard
|links| http://arxiv.org/abs/2411.05729v1 |
|updated| 2024-11-08 17:40:43 UTC |
|summary| Representing and exploiting multivariate signals require capturing complexrelations between variables. We define a novel Graph-Dictionary signal modelwhere a finite set of graphs characterizes relationships in data distributionthrough a weighted sum of their Laplacians. We propose a framework to infer thegraph dictionary representation from observed data along with a bilineargeneralization of the primal-dual splitting algorithm to solve the learningproblem. Our new formulation allows to include a priori knowledge on signalproperties as well as on underlying graphs and their coefficients. We show thecapability of our method to reconstruct graphs from signals in multiplesynthetic settings where our model outperforms previous baselines. Then weexploit graph-dictionary representations in a motor imagery decoding task onbrain activity data where we classify imagined motion better than standardmethods relying on many more features. |


| Item |Content|
| --- |---|
|idx| 2411.05661v1 |
|title| Multi-armed Bandits with Missing Outcome |
|authors| Ilia MahrooghiMahshad MoradiSina AkbariNegar Kiyavash
|links| http://arxiv.org/abs/2411.05661v1 |
|updated| 2024-11-08 16:02:39 UTC |
|summary| While significant progress has been made in designing algorithms thatminimize regret in online decision-making real-world scenarios often introduceadditional complexities perhaps the most challenging of which is missingoutcomes. Overlooking this aspect or simply assuming random missingnessinvariably leads to biased estimates of the rewards and may result in linearregret. Despite the practical relevance of this challenge no rigorousmethodology currently exists for systematically handling missingnessespecially when the missingness mechanism is not random. In this paper weaddress this gap in the context of multi-armed bandits MAB with missingoutcomes by analyzing the impact of different missingness mechanisms onachievable regret bounds. We introduce algorithms that account for missingnessunder both missing at random MAR and missing not at random MNAR models.Through both analytical and simulation studies we demonstrate the drasticimprovements in decision-making by accounting for missingness in thesesettings. |


| Item |Content|
| --- |---|
|idx| 2411.05625v1 |
|title| Cross-validating causal discovery via Leave-One-Variable-Out |
|authors| Daniela SchkodaPhilipp FallerPatrick BlöbaumDominik Janzing
|links| http://arxiv.org/abs/2411.05625v1 |
|updated| 2024-11-08 15:15:34 UTC |
|summary| We propose a new approach to falsify causal discovery algorithms withoutground truth which is based on testing the causal model on a pair of variablesthat has been dropped when learning the causal model. To this end we use theLeave-One-Variable-Out LOVO prediction where Y is inferred from Xwithout any joint observations of X and Y given only training data fromXZ_1dotsZ_k and from Z_1dotsZ_kY. We demonstrate that causal modelson the two subsets in the form of Acyclic Directed Mixed Graphs ADMGs oftenentail conclusions on the dependencies between X and Y enabling this typeof prediction. The prediction error can then be estimated since the jointdistribution PX Y is assumed to be available and X and Y have onlybeen omitted for the purpose of falsification. After presenting this graphicalmethod which is applicable to general causal discovery algorithms weillustrate how to construct a LOVO predictor tailored towards algorithmsrelying on specific a priori assumptions such as linear additive noise models.Simulations indicate that the LOVO prediction error is indeed correlated withthe accuracy of the causal outputs affirming the methods effectiveness. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2411.05783v1 |
|title| ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles |
|authors| Kayo YinChinmay SinghFyodor O. MinakovVanessa MilanHal Daumé IIICyril ZhangAlex X. LuDanielle Bragg
|links| http://arxiv.org/abs/2411.05783v1 |
|updated| 2024-11-08 18:50:37 UTC |
|summary| Deaf and hard-of-hearing DHH students face significant barriers inaccessing science technology engineering and mathematics STEM educationnotably due to the scarcity of STEM resources in signed languages. To helpaddress this we introduce ASL STEM Wiki: a parallel corpus of 254 Wikipediaarticles on STEM topics in English interpreted into over 300 hours of AmericanSign Language ASL. ASL STEM Wiki is the first continuous signing datasetfocused on STEM facilitating the development of AI resources for STEMeducation in ASL. We identify several use cases of ASL STEM Wiki withhuman-centered applications. For example because this dataset highlights thefrequent use of fingerspelling for technical concepts which inhibits DHHstudents ability to learn we develop models to identify fingerspelled words-- which can later be used to query for appropriate ASL signs to suggest tointerpreters. |


| Item |Content|
| --- |---|
|idx| 2411.05777v1 |
|title| Quantitative Assessment of Intersectional Empathetic Bias and Understanding |
|authors| Vojtech FormanekOndrej Sotolar
|links| http://arxiv.org/abs/2411.05777v1 |
|updated| 2024-11-08 18:43:15 UTC |
|summary| A growing amount of literature critiques the current operationalizations ofempathy based on loose definitions of the construct. Such definitionsnegatively affect dataset quality model robustness and evaluationreliability. We propose an empathy evaluation framework that operationalizesempathy close to its psychological origins. The framework measures the variancein responses of LLMs to prompts using existing metrics for empathy andemotional valence. The variance is introduced through the controlled generationof the prompts by varying social biases affecting context understanding thusimpacting empathetic understanding. The control over generation ensures hightheoretical validity of the constructs in the prompt dataset. Also it makeshigh-quality translation especially into languages that currently havelittle-to-no way of evaluating empathy or bias such as the Slavonic familymore manageable. Using chosen LLMs and various prompt types we demonstrate theempathy evaluation with the framework including multiple-choice answers andfree generation. The variance in our initial evaluation sample is small and wewere unable to measure convincing differences between the empatheticunderstanding in contexts given by different social groups. However theresults are promising because the models showed significant alterations theirreasoning chains needed to capture the relatively subtle changes in theprompts. This provides the basis for future research into the construction ofthe evaluation sample and statistical methods for measuring the results. |


| Item |Content|
| --- |---|
|idx| 2411.05769v1 |
|title| Effects of Distributed Friction Actuation During Sliding Touch |
|authors| MacKenzie HarnettParas KumarRebecca F. Friesen
|links| http://arxiv.org/abs/2411.05769v1 |
|updated| 2024-11-08 18:31:59 UTC |
|summary| Friction modulation allows for a range of different sensations and texturesto be simulated on flat touchscreens yet is largely unable to renderfundamental tactile interactions such as path following or shape discriminationdue to lack of spatial force distribution across the fingerpad. In order toexpand the range of sensations rendered via friction modulation in this paperwe explore the possibility of applying spatial feedback on the fingerpad viadiffering friction forces on flat touchscreens. To this end we fabricated sixdistinct flat surfaces with different spatial distributions of friction andobserved deformation of the fingerpad skin in response to motion along thesephysical samples. In our study friction changes that occur sequentially alongthe sliding direction introduced little transitory spatial warping such ascompression or stretching to the fingerpad suggesting limited perceptualdifferences in comparison to classic friction modulation. Distributingfriction across the direction of motion however showed pattern-dependentshearing of the fingertip skin opening avenues for new sensations andillusions heretofore unachievable on flat touchscreen surfaces. |


| Item |Content|
| --- |---|
|idx| 2411.05732v1 |
|title| Foundations for the psychological safety of human and autonomous vehicles interaction |
|authors| Yandika SirgabsouBenjamin HardinFrançois LeblancEfi RailiPericle SalviniDavid JacksonMarina JirotkaLars Kunze
|links| http://arxiv.org/abs/2411.05732v1 |
|updated| 2024-11-08 17:44:40 UTC |
|summary| This paper addresses the critical issue of psychological safety in the designand operation of autonomous vehicles which are increasingly integrated withartificial intelligence technologies. While traditional safety standards focusprimarily on physical safety this paper emphasizes the psychologicalimplications that arise from human interactions with autonomous vehicleshighlighting the importance of trust and perceived risk as significant factorsinfluencing user acceptance. Through a review of existing safety techniquesthe paper defines psychological safety in the context of autonomous vehiclesproposes a risk model to identify and assess psychological risks and adopts asystem-theoretic analysis method. The paper illustrates the potentialpsychological hazards using a scenario involving a familys experience with anautonomous vehicle aiming to systematically evaluate situations that couldlead to psychological harm. By establishing a framework that incorporatespsychological safety alongside physical safety the paper contributes to thebroader discourse on the safe deployment of autonomous vehicle and aims toguide future developments in user-cantered design and regulatory practices. |


| Item |Content|
| --- |---|
|idx| 2411.05653v1 |
|title| The influence of persona and conversational task on social interactions with a LLM-controlled embodied conversational agent |
|authors| Leon O. H. KroczekAlexander MaySelina HettenkoferAndreas RuiderBernd LudwigAndreas Mühlberger
|links| http://arxiv.org/abs/2411.05653v1 |
|updated| 2024-11-08 15:49:42 UTC |
|summary| Large Language Models LLMs have demonstrated remarkable capabilities inconversational tasks. Embodying an LLM as a virtual human allows users toengage in face-to-face social interactions in Virtual Reality. However theinfluence of person- and task-related factors in social interactions withLLM-controlled agents remains unclear. In this study forty-six participantsinteracted with a virtual agent whose persona was manipulated as extravert orintrovert in three different conversational tasks small talk knowledge testconvincing. Social-evaluation emotional experience and realism were assessedusing ratings. Interactive engagement was measured by quantifying participantswords and conversational turns. Finally we measured participants willingnessto ask the agent for help during the knowledge test. Our findings show that theextraverted agent was more positively evaluated elicited a more pleasantexperience and greater engagement and was assessed as more realistic comparedto the introverted agent. Whereas persona did not affect the tendency to askfor help participants were generally more confident in the answer when theyhad help of the LLM. Variation of personality traits of LLM-controlled embodiedvirtual agents therefore affects social-emotional processing and behavior invirtual interactions. Embodied virtual agents allow the presentation ofnaturalistic social encounters in a virtual environment. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2411.05683v1 |
|title| Data-Driven Distributed Common Operational Picture from Heterogeneous Platforms using Multi-Agent Reinforcement Learning |
|authors| Indranil SurAswin RaghavanAbrar RahmanJames Z HareDaniel CassentiCarl Busart
|links| http://arxiv.org/abs/2411.05683v1 |
|updated| 2024-11-08 16:31:22 UTC |
|summary| The integration of unmanned platforms equipped with advanced sensors promisesto enhance situational awareness and mitigate the fog of war in militaryoperations. However managing the vast influx of data from these platformsposes a significant challenge for Command and Control C2 systems. This studypresents a novel multi-agent learning framework to address this challenge. Ourmethod enables autonomous and secure communication between agents and humanswhich in turn enables real-time formation of an interpretable CommonOperational Picture COP. Each agent encodes its perceptions and actions intocompact vectors which are then transmitted received and decoded to form a COPencompassing the current state of all agents friendly and enemy on thebattlefield. Using Deep Reinforcement Learning DRL we jointly train COPmodels and agents action selection policies. We demonstrate resilience todegraded conditions such as denied GPS and disrupted communications.Experimental validation is performed in the Starcraft-2 simulation environmentto evaluate the precision of the COPs and robustness of policies. We reportless than 5 error in COPs and policies resilient to various adversarialconditions. In summary our contributions include a method for autonomous COPformation increased resilience through distributed prediction and jointtraining of COP models and multi-agent RL policies. This research advancesadaptive and resilient C2 facilitating effective control of heterogeneousunmanned platforms. |


| Item |Content|
| --- |---|
|idx| 2411.05599v1 |
|title| Expectation vs. Reality: Towards Verification of Psychological Games |
|authors| Marta KwiatkowskaGethin NormanDavid ParkerGabriel Santos
|links| http://arxiv.org/abs/2411.05599v1 |
|updated| 2024-11-08 14:41:52 UTC |
|summary| Game theory provides an effective way to model strategic interactions amongrational agents. In the context of formal verification these ideas can be usedto produce guarantees on the correctness of multi-agent systems with a diverserange of applications from computer security to autonomous driving.Psychological games PGs were developed as a way to model and analyse agentswith belief-dependent motivations opening up the possibility to model howhuman emotions can influence behaviour. In PGs players utilities depend notonly on what actually happens which strategies players choose to adopt butalso on what the players had expected to happen their belief as to thestrategies that would be played. Despite receiving much attention in fieldssuch as economics and psychology very little consideration has been given totheir applicability to problems in computer science nor to practicalalgorithms and tool support. In this paper we start to bridge that gapproposing methods to solve PGs and implementing them within PRISM-games aformal verification tool for stochastic games. We discuss how to model thesegames highlight specific challenges for their analysis and illustrate theusefulness of our approach on several case studies including human behaviourin traffic scenarios. |


| Item |Content|
| --- |---|
|idx| 2411.04925v2 |
|title| StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration |
|authors| Panwen HuJin JiangJianqi ChenMingfei HanShengcai LiaoXiaojun ChangXiaodan Liang
|links| http://arxiv.org/abs/2411.04925v2 |
|updated| 2024-11-11 13:24:18 UTC |
|summary| The advent of AI-Generated Content AIGC has spurred research into automatedvideo generation to streamline conventional processes. However automatingstorytelling video production particularly for customized narratives remainschallenging due to the complexity of maintaining subject consistency acrossshots. While existing approaches like Mora and AesopAgent integrate multipleagents for Story-to-Video S2V generation they fall short in preservingprotagonist consistency and supporting Customized Storytelling Video GenerationCSVG. To address these limitations we propose StoryAgent a multi-agentframework designed for CSVG. StoryAgent decomposes CSVG into distinct subtasksassigned to specialized agents mirroring the professional production process.Notably our framework includes agents for story design storyboard generationvideo creation agent coordination and result evaluation. Leveraging thestrengths of different models StoryAgent enhances control over the generationprocess significantly improving character consistency. Specifically weintroduce a customized Image-to-Video I2V method LoRA-BE to enhanceintra-shot temporal consistency while a novel storyboard generation pipelineis proposed to maintain subject consistency across shots. Extensive experimentsdemonstrate the effectiveness of our approach in synthesizing highly consistentstorytelling videos outperforming state-of-the-art methods. Our contributionsinclude the introduction of StoryAgent a versatile framework for videogeneration tasks and novel techniques for preserving protagonist consistency. |


| Item |Content|
| --- |---|
|idx| 2411.04679v1 |
|title| CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation |
|authors| Jie LiuPan ZhouYingjun DuAh-Hwee TanCees G. M. SnoekJan-Jakob SonkeEfstratios Gavves
|links| http://arxiv.org/abs/2411.04679v1 |
|updated| 2024-11-07 13:08:04 UTC |
|summary| In this work we address the cooperation problem among large language modelLLM based embodied agents where agents must cooperate to achieve a commongoal. Previous methods often execute actions extemporaneously and incoherentlywithout long-term strategic and cooperative planning leading to redundantsteps failures and even serious repercussions in complex tasks likesearch-and-rescue missions where discussion and cooperative plan are crucial.To solve this issue we propose Cooperative Plan Optimization CaPo to enhancethe cooperation efficiency of LLM-based embodied agents. Inspired by humancooperation schemes CaPo improves cooperation efficiency with two phases: 1meta-plan generation and 2 progress-adaptive meta-plan and execution. In thefirst phase all agents analyze the task discuss and cooperatively create ameta-plan that decomposes the task into subtasks with detailed steps ensuringa long-term strategic and coherent plan for efficient coordination. In thesecond phase agents execute tasks according to the meta-plan and dynamicallyadjust it based on their latest progress e.g. discovering a target objectthrough multi-turn discussions. This progress-based adaptation eliminatesredundant actions improving the overall cooperation efficiency of agents.Experimental results on the ThreeDworld Multi-Agent Transport and CommunicativeWatch-And-Help tasks demonstrate that CaPo achieves much higher task completionrate and efficiency compared with state-of-the-arts. |


| Item |Content|
| --- |---|
|idx| 2411.04678v1 |
|title| Socially-Aware Opinion-Based Navigation with Oval Limit Cycles |
|authors| Giulia d'AddatoPlacido FalquetoLuigi PalopoliDaniele Fontanelli
|links| http://arxiv.org/abs/2411.04678v1 |
|updated| 2024-11-07 13:06:16 UTC |
|summary| When humans move in a shared space they choose navigation strategies thatpreserve their mutual safety. At the same time each human seeks to minimisethe number of modifications to her/his path. In order to achieve this resulthumans use unwritten rules and reach a consensus on their decisions about themotion direction by exchanging non-verbal messages. They then implement theirchoice in a mutually acceptable way. Socially-aware navigation denotes aresearch effort aimed at replicating this logic inside robots. Existing resultsfocus either on how robots can participate in negotiations with humans or onhow they can move in a socially acceptable way. We propose a holistic approachin which the two aspects are jointly considered. Specifically we show that bycombining opinion dynamics to reach a consensus with vortex fields togenerate socially acceptable trajectories the result outperforms theapplication of the two techniques in isolation. |


