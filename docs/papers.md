# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.03218v1 |
|title| The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning |
|authors| Nathaniel LiAlexander PanAnjali GopalSummer YueDaniel BerriosAlice GattiJustin D. LiAnn-Kathrin DombrowskiShashwat GoelLong PhanGabriel MukobiNathan Helm-BurgerRassin LababidiLennart JustenAndrew B. LiuMichael ChenIsabelle BarrassOliver ZhangXiaoyuan ZhuRishub TamirisaBhrugu BharathiAdam KhojaAriel Herbert-VossCort B. BreuerAndy ZouMantas MazeikaZifan WangPalash OswalWeiran LiuAdam A. HuntJustin Tienken-HarderKevin Y. ShihKemper TalleyJohn GuanRussell KaplanIan StenekerDavid CampbellBrad JokubaitisAlex LevinsonJean WangWilliam QianKallol Krishna KarmakarSteven BasartStephen FitzMindy LevinePonnurangam KumaraguruUday TupakulaVijay VaradharajanYan ShoshitaishviliJimmy BaKevin M. EsveltAlexandr WangDan Hendrycks
|links| http://arxiv.org/abs/2403.03218v1 |
|updated| 2024-03-05 18:59:35 UTC |
|summary| The White House Executive Order on Artificial Intelligence highlights therisks of large language models LLMs empowering malicious actors in developingbiological cyber and chemical weapons. To measure these risks of malicioususe government institutions and major AI labs are developing evaluations forhazardous capabilities in LLMs. However current evaluations are privatepreventing further research into mitigating risk. Furthermore they focus ononly a few highly specific pathways for malicious use. To fill these gaps wepublicly release the Weapons of Mass Destruction Proxy WMDP benchmark adataset of 4157 multiple-choice questions that serve as a proxy measurement ofhazardous knowledge in biosecurity cybersecurity and chemical security. WMDPwas developed by a consortium of academics and technical consultants and wasstringently filtered to eliminate sensitive information prior to publicrelease. WMDP serves two roles: first as an evaluation for hazardous knowledgein LLMs and second as a benchmark for unlearning methods to remove suchhazardous knowledge. To guide progress on unlearning we develop CUT astate-of-the-art unlearning method based on controlling model representations.CUT reduces model performance on WMDP while maintaining general capabilities inareas such as biology and computer science suggesting that unlearning may be aconcrete path towards reducing malicious use from LLMs. We release ourbenchmark and code publicly at https://wmdp.ai |


| Item |Content|
| --- |---|
|idx| 2403.03194v1 |
|title| MAGID: An Automated Pipeline for Generating Synthetic Multi-modal Datasets |
|authors| Hossein AboutalebiHwanjun SongYusheng XieArshit GuptaJustin SunHang SuIgor ShalyminovNikolaos PappasSiffi SinghSaab Mansour
|links| http://arxiv.org/abs/2403.03194v1 |
|updated| 2024-03-05 18:31:28 UTC |
|summary| Development of multimodal interactive systems is hindered by the lack ofrich multimodal text images conversational data which is needed in largequantities for LLMs. Previous approaches augment textual dialogues withretrieved images posing privacy diversity and quality constraints. In thiswork we introduce textbfMultimodal textbfAugmented textbfGenerativetextbfImages textbfDialogues MAGID a framework to augment text-onlydialogues with diverse and high-quality images. Subsequently a diffusion modelis applied to craft corresponding images ensuring alignment with theidentified text. Finally MAGID incorporates an innovative feedback loopbetween an image description generation module textual LLM and image qualitymodules addressing aesthetics image-text matching and safety that work intandem to generate high-quality and multi-modal dialogues. We compare MAGID toother SOTA baselines on three dialogue datasets using automated and humanevaluation. Our results show that MAGID is comparable to or better thanbaselines with significant improvements in human evaluation especiallyagainst retrieval baselines where the image database is small. |


| Item |Content|
| --- |---|
|idx| 2403.03187v1 |
|title| Reliable, Adaptable, and Attributable Language Models with Retrieval |
|authors| Akari AsaiZexuan ZhongDanqi ChenPang Wei KohLuke ZettlemoyerHannaneh HajishirziWen-tau Yih
|links| http://arxiv.org/abs/2403.03187v1 |
|updated| 2024-03-05 18:22:33 UTC |
|summary| Parametric language models LMs which are trained on vast amounts of webdata exhibit remarkable flexibility and capability. However they still facepractical challenges such as hallucinations difficulty in adapting to new datadistributions and a lack of verifiability. In this position paper we advocatefor retrieval-augmented LMs to replace parametric LMs as the next generation ofLMs. By incorporating large-scale datastores during inferenceretrieval-augmented LMs can be more reliable adaptable and attributable.Despite their potential retrieval-augmented LMs have yet to be widely adopteddue to several obstacles: specifically current retrieval-augmented LMsstruggle to leverage helpful text beyond knowledge-intensive tasks such asquestion answering have limited interaction between retrieval and LMcomponents and lack the infrastructure for scaling. To address these wepropose a roadmap for developing general-purpose retrieval-augmented LMs. Thisinvolves a reconsideration of datastores and retrievers the exploration ofpipelines with improved retriever-LM interaction and significant investment ininfrastructure for efficient training and inference. |


| Item |Content|
| --- |---|
|idx| 2403.03170v1 |
|title| SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection |
|authors| Peng QiZehong YanWynne HsuMong Li Lee
|links| http://arxiv.org/abs/2403.03170v1 |
|updated| 2024-03-05 18:04:59 UTC |
|summary| Misinformation is a prevalent societal issue due to its potential high risks.Out-of-context OOC misinformation where authentic images are repurposed withfalse text is one of the easiest and most effective ways to mislead audiences.Current methods focus on assessing image-text consistency but lack convincingexplanations for their judgments which is essential for debunkingmisinformation. While Multimodal Large Language Models MLLMs have richknowledge and innate capability for visual reasoning and explanationgeneration they still lack sophistication in understanding and discovering thesubtle crossmodal differences. In this paper we introduce SNIFFER a novelmultimodal large language model specifically engineered for OOC misinformationdetection and explanation. SNIFFER employs two-stage instruction tuning onInstructBLIP. The first stage refines the models concept alignment of genericobjects with news-domain entities and the second stage leverages language-onlyGPT-4 generated OOC-specific instruction data to fine-tune the modelsdiscriminatory powers. Enhanced by external tools and retrieval SNIFFER notonly detects inconsistencies between text and image but also utilizes externalknowledge for contextual verification. Our experiments show that SNIFFERsurpasses the original MLLM by over 40 and outperforms state-of-the-artmethods in detection accuracy. SNIFFER also provides accurate and persuasiveexplanations as validated by quantitative and human evaluations. |


| Item |Content|
| --- |---|
|idx| 2403.03167v1 |
|title| PARADISE: Evaluating Implicit Planning Skills of Language Models with Procedural Warnings and Tips Dataset |
|authors| Arda UzunoğluAbdalfatah Rashid SafaGözde Gül Şahin
|links| http://arxiv.org/abs/2403.03167v1 |
|updated| 2024-03-05 18:01:59 UTC |
|summary| Recently there has been growing interest within the community regardingwhether large language models are capable of planning or executing plans.However most prior studies use LLMs to generate high-level plans forsimplified scenarios lacking linguistic complexity and domain diversitylimiting analysis of their planning abilities. These setups constrainevaluation methods e.g. predefined action space architectural choicese.g. only generative models and overlook the linguistic nuances essentialfor realistic analysis. To tackle this we present PARADISE an abductivereasoning task using QA format on practical procedural text sourced fromwikiHow. It involves warning and tip inference tasks directly associated withgoals excluding intermediary steps with the aim of testing the ability of themodels to infer implicit knowledge of the plan solely from the given goal. Ourexperiments utilizing fine-tuned language models and zero-shot promptingreveal the effectiveness of task-specific small models over large languagemodels in most scenarios. Despite advancements all models fall short of humanperformance. Notably our analysis uncovers intriguing insights such asvariations in model behavior with dropped keywords struggles of BERT-familyand GPT-4 with physical and abstract goals and the proposed tasks offeringvaluable prior knowledge for other unseen procedural tasks. The PARADISEdataset and associated resources are publicly available for further researchexploration with https://github.com/GGLAB-KU/paradise. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.03218v1 |
|title| The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning |
|authors| Nathaniel LiAlexander PanAnjali GopalSummer YueDaniel BerriosAlice GattiJustin D. LiAnn-Kathrin DombrowskiShashwat GoelLong PhanGabriel MukobiNathan Helm-BurgerRassin LababidiLennart JustenAndrew B. LiuMichael ChenIsabelle BarrassOliver ZhangXiaoyuan ZhuRishub TamirisaBhrugu BharathiAdam KhojaAriel Herbert-VossCort B. BreuerAndy ZouMantas MazeikaZifan WangPalash OswalWeiran LiuAdam A. HuntJustin Tienken-HarderKevin Y. ShihKemper TalleyJohn GuanRussell KaplanIan StenekerDavid CampbellBrad JokubaitisAlex LevinsonJean WangWilliam QianKallol Krishna KarmakarSteven BasartStephen FitzMindy LevinePonnurangam KumaraguruUday TupakulaVijay VaradharajanYan ShoshitaishviliJimmy BaKevin M. EsveltAlexandr WangDan Hendrycks
|links| http://arxiv.org/abs/2403.03218v1 |
|updated| 2024-03-05 18:59:35 UTC |
|summary| The White House Executive Order on Artificial Intelligence highlights therisks of large language models LLMs empowering malicious actors in developingbiological cyber and chemical weapons. To measure these risks of malicioususe government institutions and major AI labs are developing evaluations forhazardous capabilities in LLMs. However current evaluations are privatepreventing further research into mitigating risk. Furthermore they focus ononly a few highly specific pathways for malicious use. To fill these gaps wepublicly release the Weapons of Mass Destruction Proxy WMDP benchmark adataset of 4157 multiple-choice questions that serve as a proxy measurement ofhazardous knowledge in biosecurity cybersecurity and chemical security. WMDPwas developed by a consortium of academics and technical consultants and wasstringently filtered to eliminate sensitive information prior to publicrelease. WMDP serves two roles: first as an evaluation for hazardous knowledgein LLMs and second as a benchmark for unlearning methods to remove suchhazardous knowledge. To guide progress on unlearning we develop CUT astate-of-the-art unlearning method based on controlling model representations.CUT reduces model performance on WMDP while maintaining general capabilities inareas such as biology and computer science suggesting that unlearning may be aconcrete path towards reducing malicious use from LLMs. We release ourbenchmark and code publicly at https://wmdp.ai |


| Item |Content|
| --- |---|
|idx| 2403.03203v1 |
|title| CLEVR-POC: Reasoning-Intensive Visual Question Answering in Partially Observable Environments |
|authors| Savitha Sam AbrahamMarjan AlirezaieLuc De Raedt
|links| http://arxiv.org/abs/2403.03203v1 |
|updated| 2024-03-05 18:41:37 UTC |
|summary| The integration of learning and reasoning is high on the research agenda inAI. Nevertheless there is only a little attention to use existing backgroundknowledge for reasoning about partially observed scenes to answer questionsabout the scene. Yet we as humans use such knowledge frequently to inferplausible answers to visual questions by eliminating all inconsistent ones.Such knowledge often comes in the form of constraints about objects and ittends to be highly domain or environment-specific. We contribute a novelbenchmark called CLEVR-POC for reasoning-intensive visual question answeringVQA in partially observable environments under constraints. In CLEVR-POCknowledge in the form of logical constraints needs to be leveraged to generateplausible answers to questions about a hidden object in a given partial scene.For instance if one has the knowledge that all cups are colored either redgreen or blue and that there is only one green cup it becomes possible todeduce the color of an occluded cup as either red or blue provided that allother cups including the green one are observed. Through experiments weobserve that the low performance of pre-trained vision language models likeCLIP  22 and a large language model LLM like GPT-4  46 on CLEVR-POCascertains the necessity for frameworks that can handle reasoning-intensivetasks where environment-specific background knowledge is available and crucial.Furthermore our demonstration illustrates that a neuro-symbolic model whichintegrates an LLM like GPT-4 with a visual perception network and a formallogical reasoner exhibits exceptional performance on CLEVR-POC. |


| Item |Content|
| --- |---|
|idx| 2403.03188v1 |
|title| Towards Democratized Flood Risk Management: An Advanced AI Assistant Enabled by GPT-4 for Enhanced Interpretability and Public Engagement |
|authors| Rafaela MarteloRuo-Qian Wang
|links| http://arxiv.org/abs/2403.03188v1 |
|updated| 2024-03-05 18:24:52 UTC |
|summary| Real-time flood forecasting plays a crucial role in enabling timely andeffective emergency responses. However a significant challenge lies inbridging the gap between complex numerical flood models and practicaldecision-making. Decision-makers often rely on experts to interpret thesemodels for optimizing flood mitigation strategies. And the public requirescomplex techniques to inquiry and understand socio-cultural and institutionalfactors often hinders the publics understanding of flood risks. To overcomethese challenges our study introduces an innovative solution: a customized AIAssistant powered by the GPT-4 Large Language Model. This AI Assistant isdesigned to facilitate effective communication between decision-makers thegeneral public and flood forecasters without the requirement of specializedknowledge. The new framework utilizes GPT-4s advanced natural languageunderstanding and function calling capabilities to provide immediate floodalerts and respond to various flood-related inquiries. Our developed prototypeintegrates real-time flood warnings with flood maps and social vulnerabilitydata. It also effectively translates complex flood zone information intoactionable risk management advice. To assess its performance we evaluated theprototype using six criteria within three main categories: relevance errorresilience and understanding of context. Our research marks a significant steptowards a more accessible and user-friendly approach in flood risk management.This study highlights the potential of advanced AI tools like GPT-4 indemocratizing information and enhancing public engagement in critical socialand environmental issues. |


| Item |Content|
| --- |---|
|idx| 2403.03187v1 |
|title| Reliable, Adaptable, and Attributable Language Models with Retrieval |
|authors| Akari AsaiZexuan ZhongDanqi ChenPang Wei KohLuke ZettlemoyerHannaneh HajishirziWen-tau Yih
|links| http://arxiv.org/abs/2403.03187v1 |
|updated| 2024-03-05 18:22:33 UTC |
|summary| Parametric language models LMs which are trained on vast amounts of webdata exhibit remarkable flexibility and capability. However they still facepractical challenges such as hallucinations difficulty in adapting to new datadistributions and a lack of verifiability. In this position paper we advocatefor retrieval-augmented LMs to replace parametric LMs as the next generation ofLMs. By incorporating large-scale datastores during inferenceretrieval-augmented LMs can be more reliable adaptable and attributable.Despite their potential retrieval-augmented LMs have yet to be widely adopteddue to several obstacles: specifically current retrieval-augmented LMsstruggle to leverage helpful text beyond knowledge-intensive tasks such asquestion answering have limited interaction between retrieval and LMcomponents and lack the infrastructure for scaling. To address these wepropose a roadmap for developing general-purpose retrieval-augmented LMs. Thisinvolves a reconsideration of datastores and retrievers the exploration ofpipelines with improved retriever-LM interaction and significant investment ininfrastructure for efficient training and inference. |


| Item |Content|
| --- |---|
|idx| 2403.03186v1 |
|title| Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study |
|authors| Weihao TanZiluo DingWentao ZhangBoyu LiBohan ZhouJunpeng YueHaochong XiaJiechuan JiangLongtao ZhengXinrun XuYifei BiPengjie GuXinrun WangBörje F. KarlssonBo AnZongqing Lu
|links| http://arxiv.org/abs/2403.03186v1 |
|updated| 2024-03-05 18:22:29 UTC |
|summary| Recent studies have demonstrated the success of foundation agents in specifictasks or scenarios. However existing agents cannot generalize across differentscenarios mainly due to their diverse observation and action spaces andsemantic gaps or reliance on task-specific resources. In this work we proposethe General Computer Control GCC setting: building foundation agents that canmaster any computer task by taking only screen images and possibly audio ofthe computer as input and producing keyboard and mouse operations as outputsimilar to human-computer interaction. To target GCC we propose Cradle anagent framework with strong reasoning abilities including self-reflectiontask inference and skill curation to ensure generalizability andself-improvement across various tasks. To demonstrate the capabilities ofCradle we deploy it in the complex AAA game Red Dead Redemption II serving asa preliminary attempt towards GCC with a challenging target. Our agent canfollow the main storyline and finish real missions in this complex AAA gamewith minimal reliance on prior knowledge and application-specific resources.The project website is at https://baai-agents.github.io/Cradle/. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.03219v1 |
|title| LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits |
|authors| Masahiro KatoShinji Ito
|links| http://arxiv.org/abs/2403.03219v1 |
|updated| 2024-03-05 18:59:47 UTC |
|summary| This study considers the linear contextual bandit problem with independentand identically distributed i.i.d. contexts. In this problem existingstudies have proposed Best-of-Both-Worlds BoBW algorithms whose regretssatisfy Olog2T for the number of rounds T in a stochastic regime witha suboptimality gap lower-bounded by a positive constant while satisfyingOsqrtT in an adversarial regime. However the dependency on T has roomfor improvement and the suboptimality-gap assumption can be relaxed. For thisissue this study proposes an algorithm whose regret satisfies OlogT inthe setting when the suboptimality gap is lower-bounded. Furthermore weintroduce a margin condition a milder assumption on the suboptimality gap.That condition characterizes the problem difficulty linked to the suboptimalitygap using a parameter beta in 0 infty. We then show that thealgorithms regret satisfiesOleftleftlogTrightfrac1beta2betaTfrac12betaright.Here beta infty corresponds to the case in the existing studies where alower bound exists in the suboptimality gap and our regret satisfiesOlogT in that case. Our proposed algorithm is based on theFollow-The-Regularized-Leader with the Tsallis entropy and referred to as thealpha-Linear-Contextual LC-Tsallis-INF. |


| Item |Content|
| --- |---|
|idx| 2403.03218v1 |
|title| The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning |
|authors| Nathaniel LiAlexander PanAnjali GopalSummer YueDaniel BerriosAlice GattiJustin D. LiAnn-Kathrin DombrowskiShashwat GoelLong PhanGabriel MukobiNathan Helm-BurgerRassin LababidiLennart JustenAndrew B. LiuMichael ChenIsabelle BarrassOliver ZhangXiaoyuan ZhuRishub TamirisaBhrugu BharathiAdam KhojaAriel Herbert-VossCort B. BreuerAndy ZouMantas MazeikaZifan WangPalash OswalWeiran LiuAdam A. HuntJustin Tienken-HarderKevin Y. ShihKemper TalleyJohn GuanRussell KaplanIan StenekerDavid CampbellBrad JokubaitisAlex LevinsonJean WangWilliam QianKallol Krishna KarmakarSteven BasartStephen FitzMindy LevinePonnurangam KumaraguruUday TupakulaVijay VaradharajanYan ShoshitaishviliJimmy BaKevin M. EsveltAlexandr WangDan Hendrycks
|links| http://arxiv.org/abs/2403.03218v1 |
|updated| 2024-03-05 18:59:35 UTC |
|summary| The White House Executive Order on Artificial Intelligence highlights therisks of large language models LLMs empowering malicious actors in developingbiological cyber and chemical weapons. To measure these risks of malicioususe government institutions and major AI labs are developing evaluations forhazardous capabilities in LLMs. However current evaluations are privatepreventing further research into mitigating risk. Furthermore they focus ononly a few highly specific pathways for malicious use. To fill these gaps wepublicly release the Weapons of Mass Destruction Proxy WMDP benchmark adataset of 4157 multiple-choice questions that serve as a proxy measurement ofhazardous knowledge in biosecurity cybersecurity and chemical security. WMDPwas developed by a consortium of academics and technical consultants and wasstringently filtered to eliminate sensitive information prior to publicrelease. WMDP serves two roles: first as an evaluation for hazardous knowledgein LLMs and second as a benchmark for unlearning methods to remove suchhazardous knowledge. To guide progress on unlearning we develop CUT astate-of-the-art unlearning method based on controlling model representations.CUT reduces model performance on WMDP while maintaining general capabilities inareas such as biology and computer science suggesting that unlearning may be aconcrete path towards reducing malicious use from LLMs. We release ourbenchmark and code publicly at https://wmdp.ai |


| Item |Content|
| --- |---|
|idx| 2403.03208v1 |
|title| Active Statistical Inference |
|authors| Tijana ZrnicEmmanuel J. Candès
|links| http://arxiv.org/abs/2403.03208v1 |
|updated| 2024-03-05 18:46:50 UTC |
|summary| Inspired by the concept of active learning we propose activeinferenceunicodex2013a methodology for statistical inference withmachine-learning-assisted data collection. Assuming a budget on the number oflabels that can be collected the methodology uses a machine learning model toidentify which data points would be most beneficial to label thus effectivelyutilizing the budget. It operates on a simple yet powerful intuition:prioritize the collection of labels for data points where the model exhibitsuncertainty and rely on the models predictions where it is confident. Activeinference constructs provably valid confidence intervals and hypothesis testswhile leveraging any black-box machine learning model and handling any datadistribution. The key point is that it achieves the same level of accuracy withfar fewer samples than existing baselines relying on non-adaptively-collecteddata. This means that for the same number of collected samples activeinference enables smaller confidence intervals and more powerful p-values. Weevaluate active inference on datasets from public opinion research censusanalysis and proteomics. |


| Item |Content|
| --- |---|
|idx| 2403.03187v1 |
|title| Reliable, Adaptable, and Attributable Language Models with Retrieval |
|authors| Akari AsaiZexuan ZhongDanqi ChenPang Wei KohLuke ZettlemoyerHannaneh HajishirziWen-tau Yih
|links| http://arxiv.org/abs/2403.03187v1 |
|updated| 2024-03-05 18:22:33 UTC |
|summary| Parametric language models LMs which are trained on vast amounts of webdata exhibit remarkable flexibility and capability. However they still facepractical challenges such as hallucinations difficulty in adapting to new datadistributions and a lack of verifiability. In this position paper we advocatefor retrieval-augmented LMs to replace parametric LMs as the next generation ofLMs. By incorporating large-scale datastores during inferenceretrieval-augmented LMs can be more reliable adaptable and attributable.Despite their potential retrieval-augmented LMs have yet to be widely adopteddue to several obstacles: specifically current retrieval-augmented LMsstruggle to leverage helpful text beyond knowledge-intensive tasks such asquestion answering have limited interaction between retrieval and LMcomponents and lack the infrastructure for scaling. To address these wepropose a roadmap for developing general-purpose retrieval-augmented LMs. Thisinvolves a reconsideration of datastores and retrievers the exploration ofpipelines with improved retriever-LM interaction and significant investment ininfrastructure for efficient training and inference. |


| Item |Content|
| --- |---|
|idx| 2403.03185v1 |
|title| Preventing Reward Hacking with Occupancy Measure Regularization |
|authors| Cassidy LaidlawShivam SinghalAnca Dragan
|links| http://arxiv.org/abs/2403.03185v1 |
|updated| 2024-03-05 18:22:15 UTC |
|summary| Reward hacking occurs when an agent performs very well with respect to aproxy reward function which may be hand-specified or learned but poorlywith respect to the unknown true reward. Since ensuring good alignment betweenthe proxy and true reward is extremely difficult one approach to preventreward hacking is optimizing the proxy conservatively. Prior work hasparticularly focused on enforcing the learned policy to behave similarly to asafe policy by penalizing the KL divergence between their actiondistributions AD. However AD regularization doesnt always work well since asmall change in action distribution at a single state can lead to potentiallycalamitous outcomes while large changes might not be indicative of anydangerous activity. Our insight is that when reward hacking the agent visitsdrastically different states from those reached by the safe policy causinglarge deviations in state occupancy measure OM. Thus we propose regularizingbased on the OM divergence between policies instead of AD divergence to preventreward hacking. We theoretically establish that OM regularization can moreeffectively avoid large drops in true reward. Then we empirically demonstratein a variety of realistic environments that OM divergence is superior to ADdivergence for preventing reward hacking by regularizing towards a safe policy.Furthermore we show that occupancy measure divergence can also regularizelearned policies away from reward hacking behavior. Our code and data areavailable at https://github.com/cassidylaidlaw/orpo |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.03221v1 |
|title| FAR: Flexible, Accurate and Robust 6DoF Relative Camera Pose Estimation |
|authors| Chris RockwellNilesh KulkarniLinyi JinJeong Joon ParkJustin JohnsonDavid F. Fouhey
|links| http://arxiv.org/abs/2403.03221v1 |
|updated| 2024-03-05 18:59:51 UTC |
|summary| Estimating relative camera poses between images has been a central problem incomputer vision. Methods that find correspondences and solve for thefundamental matrix offer high precision in most cases. Conversely methodspredicting pose directly using neural networks are more robust to limitedoverlap and can infer absolute translation scale but at the expense of reducedprecision. We show how to combine the best of both methods our approach yieldsresults that are both precise and robust while also accurately inferringtranslation scales. At the heart of our model lies a Transformer that 1learns to balance between solved and learned pose estimations and 2 providesa prior to guide a solver. A comprehensive analysis supports our design choicesand demonstrates that our method adapts flexibly to various feature extractorsand correspondence estimators showing state-of-the-art performance in 6DoFpose estimation on Matterport3D InteriorNet StreetLearn and Map-freeRelocalization. |


| Item |Content|
| --- |---|
|idx| 2403.03217v1 |
|title| Self-supervised 3D Patient Modeling with Multi-modal Attentive Fusion |
|authors| Meng ZhengBenjamin PlancheXuan GongFan YangTerrence ChenZiyan Wu
|links| http://arxiv.org/abs/2403.03217v1 |
|updated| 2024-03-05 18:58:55 UTC |
|summary| 3D patient body modeling is critical to the success of automated patientpositioning for smart medical scanning and operating rooms. Existing CNN-basedend-to-end patient modeling solutions typically require a customized networkdesigns demanding large amount of relevant training data covering extensiverealistic clinical scenarios e.g. patient covered by sheets which leads tosuboptimal generalizability in practical deployment b expensive 3D humanmodel annotations i.e. requiring huge amount of manual effort resulting insystems that scale poorly. To address these issues we propose a genericmodularized 3D patient modeling method consists of a a multi-modal keypointdetection module with attentive fusion for 2D patient joint localization tolearn complementary cross-modality patient body information leading toimproved keypoint localization robustness and generalizability in a widevariety of imaging e.g. CT MRI etc. and clinical scenarios e.g. heavyocclusions and b a self-supervised 3D mesh regression module which does notrequire expensive 3D mesh parameter annotations to train bringing immediatecost benefits for clinical deployment. We demonstrate the efficacy of theproposed method by extensive patient positioning experiments on both public andclinical data. Our evaluation results achieve superior patient positioningperformance across various imaging modalities in real clinical scenarios. |


| Item |Content|
| --- |---|
|idx| 2403.03206v1 |
|title| Scaling Rectified Flow Transformers for High-Resolution Image Synthesis |
|authors| Patrick EsserSumith KulalAndreas BlattmannRahim EntezariJonas MüllerHarry SainiYam LeviDominik LorenzAxel SauerFrederic BoeselDustin PodellTim DockhornZion EnglishKyle LaceyAlex GoodwinYannik MarekRobin Rombach
|links| http://arxiv.org/abs/2403.03206v1 |
|updated| 2024-03-05 18:45:39 UTC |
|summary| Diffusion models create data from noise by inverting the forward paths ofdata towards noise and have emerged as a powerful generative modeling techniquefor high-dimensional perceptual data such as images and videos. Rectified flowis a recent generative model formulation that connects data and noise in astraight line. Despite its better theoretical properties and conceptualsimplicity it is not yet decisively established as standard practice. In thiswork we improve existing noise sampling techniques for training rectified flowmodels by biasing them towards perceptually relevant scales. Through alarge-scale study we demonstrate the superior performance of this approachcompared to established diffusion formulations for high-resolutiontext-to-image synthesis. Additionally we present a novel transformer-basedarchitecture for text-to-image generation that uses separate weights for thetwo modalities and enables a bidirectional flow of information between imageand text tokens improving text comprehension typography and human preferenceratings. We demonstrate that this architecture follows predictable scalingtrends and correlates lower validation loss to improved text-to-image synthesisas measured by various metrics and human evaluations. Our largest modelsoutperform state-of-the-art models and we will make our experimental datacode and model weights publicly available. |


| Item |Content|
| --- |---|
|idx| 2403.03190v2 |
|title| Triple-CFN: Restructuring Conceptual Spaces for Enhancing Abstract Reasoning process |
|authors| Ruizhuo SongBeiming Yuan
|links| http://arxiv.org/abs/2403.03190v2 |
|updated| 2024-03-06 04:21:38 UTC |
|summary| Abstract reasoning problems pose significant challenges to artificialintelligence algorithms demanding cognitive capabilities beyond those requiredfor perception tasks. This study introduces the Triple-CFN approach to tacklethe Bongard-Logo problem achieving notable reasoning accuracy by implicitlyreorganizing the concept space of conflicting instances. Additionally theTriple-CFN paradigm proves effective for the RPM problem with necessarymodifications yielding competitive results. To further enhance performance onthe RPM issue we develop the Meta Triple-CFN network which explicitlystructures the problem space while maintaining interpretability on progressivepatterns. The success of Meta Triple-CFN is attributed to its paradigm ofmodeling the conceptual space equivalent to normalizing reasoning information.Based on this ideology we introduce the Re-space layer enhancing theperformance of both Meta Triple-CFN and Triple-CFN. This paper aims tocontribute to advancements in machine intelligence by exploring innovativenetwork designs for addressing abstract reasoning problems paving the way forfurther breakthroughs in this domain. |


| Item |Content|
| --- |---|
|idx| 2403.03173v1 |
|title| Solving the bongard-logo problem by modeling a probabilistic model |
|authors| Ruizhuo SongBeiming Yuan
|links| http://arxiv.org/abs/2403.03173v1 |
|updated| 2024-03-05 18:08:29 UTC |
|summary| Abstract reasoning problems challenge the perceptual and cognitive abilitiesof AI algorithms demanding deeper pattern discernment and inductive reasoningbeyond explicit image features. This study introduces PMoC a tailoredprobability model for the Bongard-Logo problem achieving high reasoningaccuracy by constructing independent probability models. Additionally wepresent Pose-Transformer an enhanced Transformer-Encoder designed for complexabstract reasoning tasks including Bongard-Logo RAVEN I-RAVEN and PGM.Pose-Transformer incorporates positional information learning inspired bycapsule networks pose matrices enhancing its focus on local positionalrelationships in image data processing. When integrated with PMoC it furtherimproves reasoning accuracy. Our approach effectively addresses reasoningdifficulties associated with abstract entities positional changesoutperforming previous models on the OIG D3times3 subsets of RAVEN and PGMdatabases. This research contributes to advancing AIs capabilities in abstractreasoning and cognitive pattern recognition. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.03219v1 |
|title| LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits |
|authors| Masahiro KatoShinji Ito
|links| http://arxiv.org/abs/2403.03219v1 |
|updated| 2024-03-05 18:59:47 UTC |
|summary| This study considers the linear contextual bandit problem with independentand identically distributed i.i.d. contexts. In this problem existingstudies have proposed Best-of-Both-Worlds BoBW algorithms whose regretssatisfy Olog2T for the number of rounds T in a stochastic regime witha suboptimality gap lower-bounded by a positive constant while satisfyingOsqrtT in an adversarial regime. However the dependency on T has roomfor improvement and the suboptimality-gap assumption can be relaxed. For thisissue this study proposes an algorithm whose regret satisfies OlogT inthe setting when the suboptimality gap is lower-bounded. Furthermore weintroduce a margin condition a milder assumption on the suboptimality gap.That condition characterizes the problem difficulty linked to the suboptimalitygap using a parameter beta in 0 infty. We then show that thealgorithms regret satisfiesOleftleftlogTrightfrac1beta2betaTfrac12betaright.Here beta infty corresponds to the case in the existing studies where alower bound exists in the suboptimality gap and our regret satisfiesOlogT in that case. Our proposed algorithm is based on theFollow-The-Regularized-Leader with the Tsallis entropy and referred to as thealpha-Linear-Contextual LC-Tsallis-INF. |


| Item |Content|
| --- |---|
|idx| 2403.03208v1 |
|title| Active Statistical Inference |
|authors| Tijana ZrnicEmmanuel J. Candès
|links| http://arxiv.org/abs/2403.03208v1 |
|updated| 2024-03-05 18:46:50 UTC |
|summary| Inspired by the concept of active learning we propose activeinferenceunicodex2013a methodology for statistical inference withmachine-learning-assisted data collection. Assuming a budget on the number oflabels that can be collected the methodology uses a machine learning model toidentify which data points would be most beneficial to label thus effectivelyutilizing the budget. It operates on a simple yet powerful intuition:prioritize the collection of labels for data points where the model exhibitsuncertainty and rely on the models predictions where it is confident. Activeinference constructs provably valid confidence intervals and hypothesis testswhile leveraging any black-box machine learning model and handling any datadistribution. The key point is that it achieves the same level of accuracy withfar fewer samples than existing baselines relying on non-adaptively-collecteddata. This means that for the same number of collected samples activeinference enables smaller confidence intervals and more powerful p-values. Weevaluate active inference on datasets from public opinion research censusanalysis and proteomics. |


| Item |Content|
| --- |---|
|idx| 2403.03183v1 |
|title| How Well Can Transformers Emulate In-context Newton's Method? |
|authors| Angeliki GiannouLiu YangTianhao WangDimitris PapailiopoulosJason D. Lee
|links| http://arxiv.org/abs/2403.03183v1 |
|updated| 2024-03-05 18:20:10 UTC |
|summary| Transformer-based models have demonstrated remarkable in-context learningcapabilities prompting extensive research into its underlying mechanisms.Recent studies have suggested that Transformers can implement first-orderoptimization algorithms for in-context learning and even second order ones forthe case of linear regression. In this work we study whether Transformers canperform higher order optimization methods beyond the case of linearregression. We establish that linear attention Transformers with ReLU layerscan approximate second order optimization algorithms for the task of logisticregression and achieve epsilon error with only a logarithmic to the errormore layers. As a by-product we demonstrate the ability of even linearattention-only Transformers in implementing a single step of Newtons iterationfor matrix inversion with merely two layers. These results suggest the abilityof the Transformer architecture to implement complex algorithms beyondgradient descent. |


| Item |Content|
| --- |---|
|idx| 2403.03071v1 |
|title| On a Neural Implementation of Brenier's Polar Factorization |
|authors| Nina VesseronMarco Cuturi
|links| http://arxiv.org/abs/2403.03071v1 |
|updated| 2024-03-05 15:59:54 UTC |
|summary| In 1991 Brenier proved a theorem that generalizes the QR decomposition forsquare matrices -- factored as PSD times unitary -- to any vector fieldF:mathbbRdrightarrow mathbbRd. The theorem known as the polarfactorization theorem states that any field F can be recovered as thecomposition of the gradient of a convex function u with a measure-preservingmap M namely Fnabla u circ M. We propose a practical implementation ofthis far-reaching theoretical result and explore possible uses within machinelearning. The theorem is closely related to optimal transport OT theory andwe borrow from recent advances in the field of neural optimal transport toparameterize the potential u as an input convex neural network. The map Mcan be either evaluated pointwise using u the convex conjugate of uthrough the identity Mnabla u circ F or learned as an auxiliarynetwork. Because M is in general not injective we consider the additionaltask of estimating the ill-posed inverse map that can approximate the pre-imagemeasure M-1 using a stochastic generator. We illustrate possibleapplications of citeauthorBrenier1991PolarFAs polar factorization tonon-convex optimization problems as well as sampling of densities that are notlog-concave. |


| Item |Content|
| --- |---|
|idx| 2403.03069v1 |
|title| Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families |
|authors| Vaidotas SimkusMichael U. Gutmann
|links| http://arxiv.org/abs/2403.03069v1 |
|updated| 2024-03-05 15:57:52 UTC |
|summary| We consider the task of estimating variational autoencoders VAEs when thetraining data is incomplete. We show that missing data increases the complexityof the models posterior distribution over the latent variables compared to thefully-observed case. The increased complexity may adversely affect the fit ofthe model due to a mismatch between the variational and model posteriordistributions. We introduce two strategies based on i finitevariational-mixture and ii imputation-based variational-mixture distributionsto address the increased posterior complexity. Through a comprehensiveevaluation of the proposed approaches we show that variational mixtures areeffective at improving the accuracy of VAE estimation from incomplete data. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.03188v1 |
|title| Towards Democratized Flood Risk Management: An Advanced AI Assistant Enabled by GPT-4 for Enhanced Interpretability and Public Engagement |
|authors| Rafaela MarteloRuo-Qian Wang
|links| http://arxiv.org/abs/2403.03188v1 |
|updated| 2024-03-05 18:24:52 UTC |
|summary| Real-time flood forecasting plays a crucial role in enabling timely andeffective emergency responses. However a significant challenge lies inbridging the gap between complex numerical flood models and practicaldecision-making. Decision-makers often rely on experts to interpret thesemodels for optimizing flood mitigation strategies. And the public requirescomplex techniques to inquiry and understand socio-cultural and institutionalfactors often hinders the publics understanding of flood risks. To overcomethese challenges our study introduces an innovative solution: a customized AIAssistant powered by the GPT-4 Large Language Model. This AI Assistant isdesigned to facilitate effective communication between decision-makers thegeneral public and flood forecasters without the requirement of specializedknowledge. The new framework utilizes GPT-4s advanced natural languageunderstanding and function calling capabilities to provide immediate floodalerts and respond to various flood-related inquiries. Our developed prototypeintegrates real-time flood warnings with flood maps and social vulnerabilitydata. It also effectively translates complex flood zone information intoactionable risk management advice. To assess its performance we evaluated theprototype using six criteria within three main categories: relevance errorresilience and understanding of context. Our research marks a significant steptowards a more accessible and user-friendly approach in flood risk management.This study highlights the potential of advanced AI tools like GPT-4 indemocratizing information and enhancing public engagement in critical socialand environmental issues. |


| Item |Content|
| --- |---|
|idx| 2403.03101v1 |
|title| KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents |
|authors| Yuqi ZhuShuofei QiaoYixin OuShumin DengNingyu ZhangShiwei LyuYue ShenLei LiangJinjie GuHuajun Chen
|links| http://arxiv.org/abs/2403.03101v1 |
|updated| 2024-03-05 16:39:12 UTC |
|summary| Large Language Models LLMs have demonstrated great potential in complexreasoning tasks yet they fall short when tackling more sophisticatedchallenges especially when interacting with environments through generatingexecutable actions. This inadequacy primarily stems from the lack of built-inaction knowledge in language agents which fails to effectively guide theplanning trajectories during task solving and results in planninghallucination. To address this issue we introduce KnowAgent a novel approachdesigned to enhance the planning capabilities of LLMs by incorporating explicitaction knowledge. Specifically KnowAgent employs an action knowledge base anda knowledgeable self-learning strategy to constrain the action path duringplanning enabling more reasonable trajectory synthesis and thereby enhancingthe planning performance of language agents. Experimental results on HotpotQAand ALFWorld based on various backbone models demonstrate that KnowAgent canachieve comparable or superior performance to existing baselines. Furtheranalysis indicates the effectiveness of KnowAgent in terms of planninghallucinations mitigation. Code is available inhttps://github.com/zjunlp/KnowAgent. |


| Item |Content|
| --- |---|
|idx| 2403.03097v2 |
|title| Tappy: Predicting Tap Accuracy of User-Interface Elements by Reverse-Engineering Webpage Structures |
|authors| Hiroki UsubaJunichi SatoNaomi SasayaShota YamanakaFumiya Yamashita
|links| http://arxiv.org/abs/2403.03097v2 |
|updated| 2024-03-06 02:29:02 UTC |
|summary| Selecting a UI element is a fundamental operation on webpages and the easeof tapping a target object has a significant impact on usability. It is thusimportant to analyze existing UIs in order to design better ones. Howevertools proposed in previous studies cannot identify whether an element istappable on modern webpages. In this study we developed Tappy that canidentify tappable UI elements on webpages and estimate the tap-success ratebased on the element size. Our interviews of professional designers andengineers showed that Tappy helped discussions of UI design on the basis of itsquantitative metric. Furthermore we have launched this tool to be freelyavailable to external users so readers can access Tappy by visiting thewebsite https://tappy.yahoo.co.jp. |


| Item |Content|
| --- |---|
|idx| 2403.02974v1 |
|title| Online Learning of Human Constraints from Feedback in Shared Autonomy |
|authors| Shibei ZhuTran Nguyen LeSamuel KaskiVille Kyrki
|links| http://arxiv.org/abs/2403.02974v1 |
|updated| 2024-03-05 13:53:48 UTC |
|summary| Real-time collaboration with humans poses challenges due to the differentbehavior patterns of humans resulting from diverse physical constraints.Existing works typically focus on learning safety constraints forcollaboration or how to divide and distribute the subtasks between theparticipating agents to carry out the main task. In contrast we propose tolearn a human constraints model that in addition considers the diversebehaviors of different human operators. We consider a type of collaboration ina shared-autonomy fashion where both a human operator and an assistive robotact simultaneously in the same task space that affects each others actions.The task of the assistive agent is to augment the skill of humans to perform ashared task by supporting humans as much as possible both in terms of reducingthe workload and minimizing the discomfort for the human operator. Thereforewe propose an augmentative assistant agent capable of learning and adapting tohuman physical constraints aligning its actions with the ergonomic preferencesand limitations of the human operator. |


| Item |Content|
| --- |---|
|idx| 2403.02972v1 |
|title| Bodioid: philosophical reflections on the hybrid of bodies and artefacts towards post-human |
|authors| Jiang XuGang SunJingyu XuPujie Su
|links| http://arxiv.org/abs/2403.02972v1 |
|updated| 2024-03-05 13:50:25 UTC |
|summary| The advent of the post-human era has blurred the boundary between the bodyand artifacts. Further external materials and information are more deeplyintegrated into the body making emerging technology a key driving force forshaping post-human existence and promoting bodily evolution. Based on thisthis study analyses the transformation process of three technological formsnamely tools machines and cyborgs and reveals the construction of bodies andartifacts. From the phenomenological perspective the essences of body andartifact existences are reflected upon and the existence is constructionviewpoint is proposed. Furthermore a technological design concept bodioid isproposed to meticulously depict the characteristics of integrating similaritiesand differences towards unity between the body and artifacts based on thetheoretical foundation of technology mediation and the materialization ofmorality. Finally through analogizing the organizational form of language thetwo key forms and specific mechanisms of bodioid construction namely extensionand mirroring are indicated. With this in mind the post-human existencelandscape is discussed with the objective of providing theoretical insightsinto the study of the underlying philosophical principles of technologicaldesign. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.03101v1 |
|title| KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents |
|authors| Yuqi ZhuShuofei QiaoYixin OuShumin DengNingyu ZhangShiwei LyuYue ShenLei LiangJinjie GuHuajun Chen
|links| http://arxiv.org/abs/2403.03101v1 |
|updated| 2024-03-05 16:39:12 UTC |
|summary| Large Language Models LLMs have demonstrated great potential in complexreasoning tasks yet they fall short when tackling more sophisticatedchallenges especially when interacting with environments through generatingexecutable actions. This inadequacy primarily stems from the lack of built-inaction knowledge in language agents which fails to effectively guide theplanning trajectories during task solving and results in planninghallucination. To address this issue we introduce KnowAgent a novel approachdesigned to enhance the planning capabilities of LLMs by incorporating explicitaction knowledge. Specifically KnowAgent employs an action knowledge base anda knowledgeable self-learning strategy to constrain the action path duringplanning enabling more reasonable trajectory synthesis and thereby enhancingthe planning performance of language agents. Experimental results on HotpotQAand ALFWorld based on various backbone models demonstrate that KnowAgent canachieve comparable or superior performance to existing baselines. Furtheranalysis indicates the effectiveness of KnowAgent in terms of planninghallucinations mitigation. Code is available inhttps://github.com/zjunlp/KnowAgent. |


| Item |Content|
| --- |---|
|idx| 2403.03055v1 |
|title| Distributed Policy Gradient for Linear Quadratic Networked Control with Limited Communication Range |
|authors| Yuzi YanYuan Shen
|links| http://arxiv.org/abs/2403.03055v1 |
|updated| 2024-03-05 15:38:54 UTC |
|summary| This paper proposes a scalable distributed policy gradient method and provesits convergence to near-optimal solution in multi-agent linear quadraticnetworked systems. The agents engage within a specified network under localcommunication constraints implying that each agent can only exchangeinformation with a limited number of neighboring agents. On the underlyinggraph of the network each agent implements its control input depending on itsnearby neighbors states in the linear quadratic control setting. We show thatit is possible to approximate the exact gradient only using local information.Compared with the centralized optimal controller the performance gap decreasesto zero exponentially as the communication and control ranges increase. We alsodemonstrate how increasing the communication range enhances system stability inthe gradient descent process thereby elucidating a critical trade-off. Thesimulation results verify our theoretical findings. |


| Item |Content|
| --- |---|
|idx| 2403.02227v1 |
|title| Policy Space Response Oracles: A Survey |
|authors| Ariyan BighashdelYongzhao WangStephen McAleerRahul SavaniFrans A. Oliehoek
|links| http://arxiv.org/abs/2403.02227v1 |
|updated| 2024-03-04 17:15:09 UTC |
|summary| In game theory a game refers to a model of interaction among rationaldecision-makers or players making choices with the goal of achieving theirindividual objectives. Understanding their behavior in games is often referredto as game reasoning. This survey provides a comprehensive overview of afast-developing game-reasoning framework for large games known as Policy SpaceResponse Oracles PSRO. We first motivate PSRO provide historical contextand position PSRO within game-reasoning approaches. We then focus on thestrategy exploration issue for PSRO the challenge of assembling an effectivestrategy portfolio for modeling the underlying game with minimum computationalcost. We also survey current research directions for enhancing the efficiencyof PSRO and explore the applications of PSRO across various domains. Weconclude by discussing open questions and future research. |


| Item |Content|
| --- |---|
|idx| 2403.02170v1 |
|title| VITAMIN: A Compositional Framework for Model Checking of Multi-Agent Systems |
|authors| Angelo FerrandoVadim Malvone
|links| http://arxiv.org/abs/2403.02170v1 |
|updated| 2024-03-04 16:16:30 UTC |
|summary| The verification of Multi-Agent Systems MAS poses a significant challenge.Various approaches and methodologies exist to address this challenge howevertools that support them are not always readily available. Even when such toolsare accessible they tend to be hard-coded lacking in compositionality andchallenging to use due to a steep learning curve. In this paper we introduce amethodology designed for the formal verification of MAS in a modular andversatile manner along with an initial prototype that we named VITAMIN.Unlike existing verification methodologies and frameworks for MAS VITAMIN isconstructed for easy extension to accommodate various logics for specifyingthe properties to verify and models for determining on what to verify suchproperties. |


| Item |Content|
| --- |---|
|idx| 2403.02164v2 |
|title| Cognition is All You Need -- The Next Layer of AI Above Large Language Models |
|authors| Nova SpivackSam DouglasMichelle CramesTim Connors
|links| http://arxiv.org/abs/2403.02164v2 |
|updated| 2024-03-05 10:23:52 UTC |
|summary| Recent studies of the applications of conversational AI tools such aschatbots powered by large language models to complex real-world knowledge workhave shown limitations related to reasoning and multi-step problem solving.Specifically while existing chatbots simulate shallow reasoning andunderstanding they are prone to errors as problem complexity increases. Thefailure of these systems to address complex knowledge work is due to the factthat they do not perform any actual cognition. In this position paper wepresent Cognitive AI a higher-level framework for implementingprogrammatically defined neuro-symbolic cognition above and outside of largelanguage models. Specifically we propose a dual-layer functional architecturefor Cognitive AI that serves as a roadmap for AI systems that can performcomplex multi-step knowledge work. We propose that Cognitive AI is a necessaryprecursor for the evolution of higher forms of AI such as AGI andspecifically claim that AGI cannot be achieved by probabilistic approaches ontheir own. We conclude with a discussion of the implications for large languagemodels adoption cycles in AI and commercial Cognitive AI development. |


