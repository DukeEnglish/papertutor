# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.04253v1 |
|title| AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls |
|authors| Yu DuFangyun WeiHongyang Zhang
|links| http://arxiv.org/abs/2402.04253v1 |
|updated| 2024-02-06 18:59:57 UTC |
|summary| We introduce AnyTool a large language model agent designed to revolutionizethe utilization of a vast array of tools in addressing user queries. We utilizeover 16000 APIs from Rapid API operating under the assumption that a subsetof these APIs could potentially resolve the queries. AnyTool primarilyincorporates three elements: an API retriever with a hierarchical structure asolver aimed at resolving user queries using a selected set of API candidatesand a self-reflection mechanism which re-activates AnyTool if the initialsolution proves impracticable. AnyTool is powered by the function callingfeature of GPT-4 eliminating the need for training external modules. We alsorevisit the evaluation protocol introduced by previous works and identify alimitation in this protocol that leads to an artificially high pass rate. Byrevising the evaluation protocol to better reflect practical applicationscenarios we introduce an additional benchmark termed AnyToolBench.Experiments across various datasets demonstrate the superiority of our AnyToolover strong baselines such as ToolLLM and a GPT-4 variant tailored for toolutilization. For instance AnyTool outperforms ToolLLM by 35.4 in terms ofaverage pass rate on ToolBench. Code will be available athttps://github.com/dyabel/AnyTool. |


| Item |Content|
| --- |---|
|idx| 2402.04251v1 |
|title| Linear-time Minimum Bayes Risk Decoding with Reference Aggregation |
|authors| Jannis VamvasRico Sennrich
|links| http://arxiv.org/abs/2402.04251v1 |
|updated| 2024-02-06 18:59:30 UTC |
|summary| Minimum Bayes Risk MBR decoding is a text generation technique that hasbeen shown to improve the quality of machine translations but is expensiveeven if a sampling-based approximation is used. Besides requiring a largenumber of sampled sequences it requires the pairwise calculation of a utilitymetric which has quadratic complexity. In this paper we propose toapproximate pairwise metric scores with scores calculated against aggregatedreference representations. This changes the complexity of utility estimationfrom On2 to On while empirically preserving most of the quality gainsof MBR decoding. We release our source code at https://github.com/ZurichNLP/mbr |


| Item |Content|
| --- |---|
|idx| 2402.04249v1 |
|title| HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal |
|authors| Mantas MazeikaLong PhanXuwang YinAndy ZouZifan WangNorman MuElham SakhaeeNathaniel LiSteven BasartBo LiDavid ForsythDan Hendrycks
|links| http://arxiv.org/abs/2402.04249v1 |
|updated| 2024-02-06 18:59:08 UTC |
|summary| Automated red teaming holds substantial promise for uncovering and mitigatingthe risks associated with the malicious use of large language models LLMsyet the field lacks a standardized evaluation framework to rigorously assessnew methods. To address this issue we introduce HarmBench a standardizedevaluation framework for automated red teaming. We identify several desirableproperties previously unaccounted for in red teaming evaluations andsystematically design HarmBench to meet these criteria. Using HarmBench weconduct a large-scale comparison of 18 red teaming methods and 33 target LLMsand defenses yielding novel insights. We also introduce a highly efficientadversarial training method that greatly enhances LLM robustness across a widerange of attacks demonstrating how HarmBench enables codevelopment of attacksand defenses. We open source HarmBench athttps://github.com/centerforaisafety/HarmBench. |


| Item |Content|
| --- |---|
|idx| 2402.04247v2 |
|title| Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science |
|authors| Xiangru TangQiao JinKunlun ZhuTongxin YuanYichi ZhangWangchunshu ZhouMeng QuYilun ZhaoJian TangZhuosheng ZhangArman CohanZhiyong LuMark Gerstein
|links| http://arxiv.org/abs/2402.04247v2 |
|updated| 2024-02-07 14:26:02 UTC |
|summary| Intelligent agents powered by large language models LLMs have demonstratedsubstantial promise in autonomously conducting experiments and facilitatingscientific discoveries across various disciplines. While their capabilities arepromising they also introduce novel vulnerabilities that demand carefulconsideration for safety. However there exists a notable gap in theliterature as there has been no comprehensive exploration of thesevulnerabilities. This position paper fills this gap by conducting a thoroughexamination of vulnerabilities in LLM-based agents within scientific domainsshedding light on potential risks associated with their misuse and emphasizingthe need for safety measures. We begin by providing a comprehensive overview ofthe potential risks inherent to scientific LLM agents taking into account userintent the specific scientific domain and their potential impact on theexternal environment. Then we delve into the origins of these vulnerabilitiesand provide a scoping review of the limited existing works. Based on ouranalysis we propose a triadic framework involving human regulation agentalignment and an understanding of environmental feedback agent regulation tomitigate these identified risks. Furthermore we highlight the limitations andchallenges associated with safeguarding scientific agents and advocate for thedevelopment of improved models robust benchmarks and comprehensiveregulations to address these issues effectively. |


| Item |Content|
| --- |---|
|idx| 2402.04236v1 |
|title| CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations |
|authors| Ji QiMing DingWeihan WangYushi BaiQingsong LvWenyi HongBin XuLei HouJuanzi LiYuxiao DongJie Tang
|links| http://arxiv.org/abs/2402.04236v1 |
|updated| 2024-02-06 18:43:48 UTC |
|summary| Vision-Language Models VLMs have demonstrated their widespread viabilitythanks to extensive training in aligning visual instructions to answers.However this conclusive alignment leads models to ignore critical visualreasoning and further result in failures on meticulous visual problems andunfaithful responses. In this paper we propose Chain of Manipulations amechanism that enables VLMs to solve problems with a series of manipulationswhere each manipulation refers to an operation on the visual input either fromintrinsic abilities e.g. grounding acquired through prior training or fromimitating human-like behaviors e.g. zoom in. This mechanism encourages VLMsto generate faithful responses with evidential visual reasoning and permitsusers to trace error causes in the interpretable paths. We thus train CogCoM ageneral 17B VLM with a memory-based compatible architecture endowed thisreasoning mechanism. Experiments show that our model achieves thestate-of-the-art performance across 8 benchmarks from 3 categories and alimited number of training steps with the data swiftly gains a competitiveperformance. The code and data are publicly available athttps://github.com/THUDM/CogCoM. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.04249v1 |
|title| HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal |
|authors| Mantas MazeikaLong PhanXuwang YinAndy ZouZifan WangNorman MuElham SakhaeeNathaniel LiSteven BasartBo LiDavid ForsythDan Hendrycks
|links| http://arxiv.org/abs/2402.04249v1 |
|updated| 2024-02-06 18:59:08 UTC |
|summary| Automated red teaming holds substantial promise for uncovering and mitigatingthe risks associated with the malicious use of large language models LLMsyet the field lacks a standardized evaluation framework to rigorously assessnew methods. To address this issue we introduce HarmBench a standardizedevaluation framework for automated red teaming. We identify several desirableproperties previously unaccounted for in red teaming evaluations andsystematically design HarmBench to meet these criteria. Using HarmBench weconduct a large-scale comparison of 18 red teaming methods and 33 target LLMsand defenses yielding novel insights. We also introduce a highly efficientadversarial training method that greatly enhances LLM robustness across a widerange of attacks demonstrating how HarmBench enables codevelopment of attacksand defenses. We open source HarmBench athttps://github.com/centerforaisafety/HarmBench. |


| Item |Content|
| --- |---|
|idx| 2402.04247v2 |
|title| Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science |
|authors| Xiangru TangQiao JinKunlun ZhuTongxin YuanYichi ZhangWangchunshu ZhouMeng QuYilun ZhaoJian TangZhuosheng ZhangArman CohanZhiyong LuMark Gerstein
|links| http://arxiv.org/abs/2402.04247v2 |
|updated| 2024-02-07 14:26:02 UTC |
|summary| Intelligent agents powered by large language models LLMs have demonstratedsubstantial promise in autonomously conducting experiments and facilitatingscientific discoveries across various disciplines. While their capabilities arepromising they also introduce novel vulnerabilities that demand carefulconsideration for safety. However there exists a notable gap in theliterature as there has been no comprehensive exploration of thesevulnerabilities. This position paper fills this gap by conducting a thoroughexamination of vulnerabilities in LLM-based agents within scientific domainsshedding light on potential risks associated with their misuse and emphasizingthe need for safety measures. We begin by providing a comprehensive overview ofthe potential risks inherent to scientific LLM agents taking into account userintent the specific scientific domain and their potential impact on theexternal environment. Then we delve into the origins of these vulnerabilitiesand provide a scoping review of the limited existing works. Based on ouranalysis we propose a triadic framework involving human regulation agentalignment and an understanding of environmental feedback agent regulation tomitigate these identified risks. Furthermore we highlight the limitations andchallenges associated with safeguarding scientific agents and advocate for thedevelopment of improved models robust benchmarks and comprehensiveregulations to address these issues effectively. |


| Item |Content|
| --- |---|
|idx| 2402.04232v2 |
|title| Can Generative Agents Predict Emotion? |
|authors| Ciaran ReganNanami IwahashiShogo TanakaMizuki Oka
|links| http://arxiv.org/abs/2402.04232v2 |
|updated| 2024-02-07 17:27:09 UTC |
|summary| Large Language Models LLMs have demonstrated a number of human-likeabilities however the empathic understanding and emotional state of LLMs isyet to be aligned to that of humans. In this work we investigate how theemotional state of generative LLM agents evolves as they perceive new eventsintroducing a novel architecture in which new experiences are compared to pastmemories. Through this comparison the agent gains the ability to understandnew experiences in context which according to the appraisal theory of emotionis vital in emotion creation. First the agent perceives new experiences astime series text data. After perceiving each new input the agent generates asummary of past relevant memories referred to as the norm and compares thenew experience to this norm. Through this comparison we can analyse how theagent reacts to the new experience in context. The PANAS a test of affect isadministered to the agent capturing the emotional state of the agent after theperception of the new event. Finally the new experience is then added to theagents memory to be used in the creation of future norms. By creating multipleexperiences in natural language from emotionally charged situations we testthe proposed architecture on a wide range of scenarios. The mixed resultssuggests that introducing context can occasionally improve the emotionalalignment of the agent but further study and comparison with human evaluatorsis necessary. We hope that this paper is another step towards the alignment ofgenerative agents. |


| Item |Content|
| --- |---|
|idx| 2402.04228v1 |
|title| Intelligent Collective Escape of Swarm Robots Based on a Novel Fish-inspired Self-adaptive Approach with Neurodynamic Models |
|authors| Junfei LiSimon X. Yang
|links| http://dx.doi.org/10.1109/TIE.2024.3363723 |
|updated| 2024-02-06 18:36:44 UTC |
|summary| Fish schools present high-efficiency group behaviors through simpleindividual interactions to collective migration and dynamic escape from thepredator. The school behavior of fish is usually a good inspiration to designcontrol architecture for swarm robots. In this paper a novel fish-inspiredself-adaptive approach is proposed for collective escape for the swarm robots.In addition a bio-inspired neural network BINN is introduced to generatecollision-free escape robot trajectories through the combination of attractiveand repulsive forces. Furthermore to cope with dynamic environments aneurodynamics-based self-adaptive mechanism is proposed to improve theself-adaptive performance of the swarm robots in the changing environment.Similar to fish escape maneuvers simulation and experimental results show thatthe swarm robots are capable of collectively leaving away from the threats.Several comparison studies demonstrated that the proposed approach cansignificantly improve the effectiveness and efficiency of system performanceand the flexibility and robustness in complex environments. |


| Item |Content|
| --- |---|
|idx| 2402.04210v1 |
|title| "Task Success" is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors |
|authors| Lin GuanYifan ZhouDenis LiuYantian ZhaHeni Ben AmorSubbarao Kambhampati
|links| http://arxiv.org/abs/2402.04210v1 |
|updated| 2024-02-06 18:07:43 UTC |
|summary| Large-scale generative models are shown to be useful for sampling meaningfulcandidate solutions yet they often overlook task constraints and userpreferences. Their full power is better harnessed when the models are coupledwith external verifiers and the final solutions are derived iteratively orprogressively according to the verification feedback. In the context ofembodied AI verification often solely involves assessing whether goalconditions specified in the instructions have been met. Nonetheless for theseagents to be seamlessly integrated into daily life it is crucial to accountfor a broader range of constraints and preferences beyond bare task successe.g. a robot should grasp bread with care to avoid significant deformations.However given the unbounded scope of robot tasks it is infeasible toconstruct scripted verifiers akin to those used for explicit-knowledge taskslike the game of Go and theorem proving. This begs the question: when no soundverifier is available can we use large vision and language models VLMswhich are approximately omniscient as scalable Behavior Critics to catchundesirable robot behaviors in videos To answer this we first construct abenchmark that contains diverse cases of goal-reaching yet undesirable robotpolicies. Then we comprehensively evaluate VLM critics to gain a deeperunderstanding of their strengths and failure modes. Based on the evaluation weprovide guidelines on how to effectively utilize VLM critiques and showcase apractical way to integrate the feedback into an iterative process of policyrefinement. The dataset and codebase are released at:https://guansuns.github.io/pages/vlm-critic. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.04249v1 |
|title| HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal |
|authors| Mantas MazeikaLong PhanXuwang YinAndy ZouZifan WangNorman MuElham SakhaeeNathaniel LiSteven BasartBo LiDavid ForsythDan Hendrycks
|links| http://arxiv.org/abs/2402.04249v1 |
|updated| 2024-02-06 18:59:08 UTC |
|summary| Automated red teaming holds substantial promise for uncovering and mitigatingthe risks associated with the malicious use of large language models LLMsyet the field lacks a standardized evaluation framework to rigorously assessnew methods. To address this issue we introduce HarmBench a standardizedevaluation framework for automated red teaming. We identify several desirableproperties previously unaccounted for in red teaming evaluations andsystematically design HarmBench to meet these criteria. Using HarmBench weconduct a large-scale comparison of 18 red teaming methods and 33 target LLMsand defenses yielding novel insights. We also introduce a highly efficientadversarial training method that greatly enhances LLM robustness across a widerange of attacks demonstrating how HarmBench enables codevelopment of attacksand defenses. We open source HarmBench athttps://github.com/centerforaisafety/HarmBench. |


| Item |Content|
| --- |---|
|idx| 2402.04248v1 |
|title| Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks |
|authors| Jongho ParkJaeseung ParkZheyang XiongNayoung LeeJaewoong ChoSamet OymakKangwook LeeDimitris Papailiopoulos
|links| http://arxiv.org/abs/2402.04248v1 |
|updated| 2024-02-06 18:56:35 UTC |
|summary| State-space models SSMs such as Mamba Gu  Dao 2034 have been proposedas alternatives to Transformer networks in language modeling by incorporatinggating convolutions and input-dependent token selection to mitigate thequadratic cost of multi-head attention. Although SSMs exhibit competitiveperformance their in-context learning ICL capabilities a remarkableemergent property of modern language models that enables task execution withoutparameter optimization remain underexplored compared to Transformers. In thisstudy we evaluate the ICL performance of SSMs focusing on Mamba againstTransformer models across various tasks. Our results show that SSMs performcomparably to Transformers in standard regression ICL tasks whileoutperforming them in tasks like sparse parity learning. However SSMs fallshort in tasks involving non-standard retrieval functionality. To address theselimitations we introduce a hybrid model variant that combines Mamba withattention blocks surpassing individual models in tasks where they struggleindependently. Our findings suggest that hybrid architectures offer promisingavenues for enhancing ICL in language models. |


| Item |Content|
| --- |---|
|idx| 2402.04247v2 |
|title| Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science |
|authors| Xiangru TangQiao JinKunlun ZhuTongxin YuanYichi ZhangWangchunshu ZhouMeng QuYilun ZhaoJian TangZhuosheng ZhangArman CohanZhiyong LuMark Gerstein
|links| http://arxiv.org/abs/2402.04247v2 |
|updated| 2024-02-07 14:26:02 UTC |
|summary| Intelligent agents powered by large language models LLMs have demonstratedsubstantial promise in autonomously conducting experiments and facilitatingscientific discoveries across various disciplines. While their capabilities arepromising they also introduce novel vulnerabilities that demand carefulconsideration for safety. However there exists a notable gap in theliterature as there has been no comprehensive exploration of thesevulnerabilities. This position paper fills this gap by conducting a thoroughexamination of vulnerabilities in LLM-based agents within scientific domainsshedding light on potential risks associated with their misuse and emphasizingthe need for safety measures. We begin by providing a comprehensive overview ofthe potential risks inherent to scientific LLM agents taking into account userintent the specific scientific domain and their potential impact on theexternal environment. Then we delve into the origins of these vulnerabilitiesand provide a scoping review of the limited existing works. Based on ouranalysis we propose a triadic framework involving human regulation agentalignment and an understanding of environmental feedback agent regulation tomitigate these identified risks. Furthermore we highlight the limitations andchallenges associated with safeguarding scientific agents and advocate for thedevelopment of improved models robust benchmarks and comprehensiveregulations to address these issues effectively. |


| Item |Content|
| --- |---|
|idx| 2402.04239v1 |
|title| CAST: Clustering Self-Attention using Surrogate Tokens for Efficient Transformers |
|authors| Adjorn van EngelenhovenNicola StrisciuglioEstefanía Talavera
|links| http://arxiv.org/abs/2402.04239v1 |
|updated| 2024-02-06 18:47:52 UTC |
|summary| The Transformer architecture has shown to be a powerful tool for a wide rangeof tasks. It is based on the self-attention mechanism which is an inherentlycomputationally expensive operation with quadratic computational complexity:memory usage and compute time increase quadratically with the length of theinput sequences thus limiting the application of Transformers. In this workwe propose a novel Clustering self-Attention mechanism using Surrogate TokensCAST to optimize the attention computation and achieve efficienttransformers. CAST utilizes learnable surrogate tokens to construct a clusteraffinity matrix used to cluster the input sequence and generate novel clustersummaries. The self-attention from within each cluster is then combined withthe cluster summaries of other clusters enabling information flow across theentire input sequence. CAST improves efficiency by reducing the complexity fromON2 to Oalpha N where N is the sequence length and alpha isconstant according to the number of clusters and samples per cluster. We showthat CAST performs better than or comparable to the baseline Transformers onlong-range sequence modeling tasks while also achieving higher results on timeand memory efficiency than other efficient transformers. |


| Item |Content|
| --- |---|
|idx| 2402.04229v1 |
|title| MusicRL: Aligning Music Generation to Human Preferences |
|authors| Geoffrey CideronSertan GirginMauro VerzettiDamien VincentMatej KastelicZalán BorsosBrian McWilliamsVictor UngureanuOlivier BachemOlivier PietquinMatthieu GeistLéonard HussenotNeil ZeghidourAndrea Agostinelli
|links| http://arxiv.org/abs/2402.04229v1 |
|updated| 2024-02-06 18:36:52 UTC |
|summary| We propose MusicRL the first music generation system finetuned from humanfeedback. Appreciation of text-to-music models is particularly subjective sincethe concept of musicality as well as the specific intention behind a captionare user-dependent e.g. a caption such as upbeat work-out music can map to aretro guitar solo or a techno pop beat. Not only this makes supervisedtraining of such models challenging but it also calls for integratingcontinuous human feedback in their post-deployment finetuning. MusicRL is apretrained autoregressive MusicLM Agostinelli et al. 2023 model of discreteaudio tokens finetuned with reinforcement learning to maximise sequence-levelrewards. We design reward functions related specifically to text-adherence andaudio quality with the help from selected raters and use those to finetuneMusicLM into MusicRL-R. We deploy MusicLM to users and collect a substantialdataset comprising 300000 pairwise preferences. Using Reinforcement Learningfrom Human Feedback RLHF we train MusicRL-U the first text-to-music modelthat incorporates human feedback at scale. Human evaluations show that bothMusicRL-R and MusicRL-U are preferred to the baseline. Ultimately MusicRL-RUcombines the two approaches and results in the best model according to humanraters. Ablation studies shed light on the musical attributes influencing humanpreferences indicating that text adherence and quality only account for a partof it. This underscores the prevalence of subjectivity in musical appreciationand calls for further involvement of human listeners in the finetuning of musicgeneration models. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.04252v1 |
|title| EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters |
|authors| Quan SunJinsheng WangQiying YuYufeng CuiFan ZhangXiaosong ZhangXinlong Wang
|links| http://arxiv.org/abs/2402.04252v1 |
|updated| 2024-02-06 18:59:48 UTC |
|summary| Scaling up contrastive language-image pretraining CLIP is critical forempowering both vision and multimodal models. We present EVA-CLIP-18B thelargest and most powerful open-source CLIP model to date with 18-billionparameters. With only 6-billion training samples seen EVA-CLIP-18B achieves anexceptional 80.7 zero-shot top-1 accuracy averaged across 27 widely recognizedimage classification benchmarks outperforming its forerunner EVA-CLIP5-billion parameters and other open-source CLIP models by a large margin.Remarkably we observe a consistent performance improvement with the model sizescaling of EVA-CLIP despite maintaining a constant training dataset of2-billion image-text pairs from LAION-2B and COYO-700M. This dataset is openlyavailable and much smaller than the in-house datasets e.g. DFN-5B WebLI-10Bemployed in other state-of-the-art CLIP models. EVA-CLIP-18B demonstrates thepotential of EVA-style weak-to-strong visual model scaling. With our modelweights made publicly available we hope to facilitate future research invision and multimodal foundation models. |


| Item |Content|
| --- |---|
|idx| 2402.04249v1 |
|title| HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal |
|authors| Mantas MazeikaLong PhanXuwang YinAndy ZouZifan WangNorman MuElham SakhaeeNathaniel LiSteven BasartBo LiDavid ForsythDan Hendrycks
|links| http://arxiv.org/abs/2402.04249v1 |
|updated| 2024-02-06 18:59:08 UTC |
|summary| Automated red teaming holds substantial promise for uncovering and mitigatingthe risks associated with the malicious use of large language models LLMsyet the field lacks a standardized evaluation framework to rigorously assessnew methods. To address this issue we introduce HarmBench a standardizedevaluation framework for automated red teaming. We identify several desirableproperties previously unaccounted for in red teaming evaluations andsystematically design HarmBench to meet these criteria. Using HarmBench weconduct a large-scale comparison of 18 red teaming methods and 33 target LLMsand defenses yielding novel insights. We also introduce a highly efficientadversarial training method that greatly enhances LLM robustness across a widerange of attacks demonstrating how HarmBench enables codevelopment of attacksand defenses. We open source HarmBench athttps://github.com/centerforaisafety/HarmBench. |


| Item |Content|
| --- |---|
|idx| 2402.04236v1 |
|title| CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations |
|authors| Ji QiMing DingWeihan WangYushi BaiQingsong LvWenyi HongBin XuLei HouJuanzi LiYuxiao DongJie Tang
|links| http://arxiv.org/abs/2402.04236v1 |
|updated| 2024-02-06 18:43:48 UTC |
|summary| Vision-Language Models VLMs have demonstrated their widespread viabilitythanks to extensive training in aligning visual instructions to answers.However this conclusive alignment leads models to ignore critical visualreasoning and further result in failures on meticulous visual problems andunfaithful responses. In this paper we propose Chain of Manipulations amechanism that enables VLMs to solve problems with a series of manipulationswhere each manipulation refers to an operation on the visual input either fromintrinsic abilities e.g. grounding acquired through prior training or fromimitating human-like behaviors e.g. zoom in. This mechanism encourages VLMsto generate faithful responses with evidential visual reasoning and permitsusers to trace error causes in the interpretable paths. We thus train CogCoM ageneral 17B VLM with a memory-based compatible architecture endowed thisreasoning mechanism. Experiments show that our model achieves thestate-of-the-art performance across 8 benchmarks from 3 categories and alimited number of training steps with the data swiftly gains a competitiveperformance. The code and data are publicly available athttps://github.com/THUDM/CogCoM. |


| Item |Content|
| --- |---|
|idx| 2402.04195v1 |
|title| Instance by Instance: An Iterative Framework for Multi-instance 3D Registration |
|authors| Xinyue CaoXiyu ZhangYuxin ChengZhaoshuai QiYanning ZhangJiaqi Yang
|links| http://arxiv.org/abs/2402.04195v1 |
|updated| 2024-02-06 17:50:30 UTC |
|summary| Multi-instance registration is a challenging problem in computer vision androbotics where multiple instances of an object need to be registered in astandard coordinate system. In this work we propose the first iterativeframework called instance-by-instance IBI for multi-instance 3D registrationMI-3DReg. It successively registers all instances in a given scenariostarting from the easiest and progressing to more challenging ones. Throughoutthe iterative process outliers are eliminated continuously leading to anincreasing inlier rate for the remaining and more challenging instances. Underthe IBI framework we further propose a sparse-to-dense-correspondence-basedmulti-instance registration method IBI-S2DC to achieve robust MI-3DReg.Experiments on the synthetic and real datasets have demonstrated theeffectiveness of IBI and suggested the new state-of-the-art performance ofIBI-S2DC e.g. our MHF1 is 12.02/12.35 higher than the existingstate-of-the-art method ECC on the synthetic/real datasets. |


| Item |Content|
| --- |---|
|idx| 2402.04178v1 |
|title| SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models |
|authors| Yichen ShiYuhao GaoYingxin LaiHongyang WangJun FengLei HeJun WanChangsheng ChenZitong YuXiaochun Cao
|links| http://arxiv.org/abs/2402.04178v1 |
|updated| 2024-02-06 17:31:36 UTC |
|summary| Multimodal large language models MLLMs have demonstrated remarkableproblem-solving capabilities in various vision fields e.g. generic objectrecognition and grounding based on strong visual semantic representation andlanguage reasoning ability. However whether MLLMs are sensitive to subtlevisual spoof/forged clues and how they perform in the domain of face attackdetection e.g. face spoofing and forgery detection is still unexplored. Inthis paper we introduce a new benchmark namely SHIELD to evaluate theability of MLLMs on face spoofing and forgery detection. Specifically wedesign true/false and multiple-choice questions to evaluate multimodal facedata in these two face security tasks. For the face anti-spoofing task weevaluate three different modalities i.e. RGB infrared depth under fourtypes of presentation attacks i.e. print attack replay attack rigid maskpaper mask. For the face forgery detection task we evaluate GAN-based anddiffusion-based data with both visual and acoustic modalities. Each question issubjected to both zero-shot and few-shot tests under standard and chain ofthought COT settings. The results indicate that MLLMs hold substantialpotential in the face security domain offering advantages over traditionalspecific models in terms of interpretability multimodal flexible reasoningand joint face spoof and forgery detection. Additionally we develop a novelMulti-Attribute Chain of Thought MA-COT paradigm for describing and judgingvarious task-specific and task-irrelevant attributes of face images whichprovides rich task-related knowledge for subtle spoof/forged clue mining.Extensive experiments in separate face anti-spoofing separate face forgerydetection and joint detection tasks demonstrate the effectiveness of theproposed MA-COT. The project is available athttps://github.com/laiyingxin2/SHIELD |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.04211v1 |
|title| Variational Shapley Network: A Probabilistic Approach to Self-Explaining Shapley values with Uncertainty Quantification |
|authors| Mert KetenciIñigo UrteagaVictor Alfonso RodriguezNoémie ElhadadAdler Perotte
|links| http://arxiv.org/abs/2402.04211v1 |
|updated| 2024-02-06 18:09:05 UTC |
|summary| Shapley values have emerged as a foundational tool in machine learning MLfor elucidating model decision-making processes. Despite their widespreadadoption and unique ability to satisfy essential explainability axiomscomputational challenges persist in their estimation when i evaluating amodel over all possible subset of input feature combinations ii estimatingmodel marginals and iii addressing variability in explanations. Weintroduce a novel self-explaining method that simplifies the computation ofShapley values significantly requiring only a single forward pass. Recognizingthe deterministic treatment of Shapley values as a limitation we exploreincorporating a probabilistic framework to capture the inherent uncertainty inexplanations. Unlike alternatives our technique does not rely directly on theobserved data space to estimate marginals instead it uses adaptable baselinevalues derived from a latent feature-specific embedding space generated by anovel masked neural network architecture. Evaluations on simulated and realdatasets underscore our techniques robust predictive and explanatoryperformance. |


| Item |Content|
| --- |---|
|idx| 2402.04177v1 |
|title| Scaling Laws for Downstream Task Performance of Large Language Models |
|authors| Berivan IsikNatalia PonomarevaHussein HazimehDimitris PaparasSergei VassilvitskiiSanmi Koyejo
|links| http://arxiv.org/abs/2402.04177v1 |
|updated| 2024-02-06 17:31:20 UTC |
|summary| Scaling laws provide important insights that can guide the design of largelanguage models LLMs. Existing work has primarily focused on studying scalinglaws for pretraining upstream loss. However in transfer learning settingsin which LLMs are pretrained on an unsupervised dataset and then finetuned on adownstream task we often also care about the downstream performance. In thiswork we study the scaling behavior in a transfer learning setting where LLMsare finetuned for machine translation tasks. Specifically we investigate howthe choice of the pretraining data and its size affect downstream performancetranslation quality as judged by two metrics: downstream cross-entropy andBLEU score. Our experiments indicate that the size of the finetuning datasetand the distribution alignment between the pretraining and downstream datasignificantly influence the scaling behavior. With sufficient alignment bothdownstream cross-entropy and BLEU score improve monotonically with morepretraining data. In such cases we show that it is possible to predict thedownstream BLEU score with good accuracy using a log-law. However there arealso cases where moderate misalignment causes the BLEU score to fluctuate orget worse with more pretraining whereas downstream cross-entropy monotonicallyimproves. By analyzing these observations we provide new practical insightsfor choosing appropriate pretraining data. |


| Item |Content|
| --- |---|
|idx| 2402.04161v1 |
|title| Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains |
|authors| Ashok Vardhan MakkuvaMarco BondaschiAdway GirishAlliot NagleMartin JaggiHyeji KimMichael Gastpar
|links| http://arxiv.org/abs/2402.04161v1 |
|updated| 2024-02-06 17:18:59 UTC |
|summary| In recent years attention-based transformers have achieved tremendoussuccess across a variety of disciplines including natural languages. A keyingredient behind their success is the generative pretraining procedure duringwhich these models are trained on a large text corpus in an auto-regressivemanner. To shed light on this phenomenon we propose a new framework thatallows both theory and systematic experiments to study the sequential modelingcapabilities of transformers through the lens of Markov chains. Inspired by theMarkovianity of natural languages we model the data as a Markovian source andutilize this framework to systematically study the interplay between thedata-distributional properties the transformer architecture the learntdistribution and the final model performance. In particular we theoreticallycharacterize the loss landscape of single-layer transformers and show theexistence of global minima and bad local minima contingent upon the specificdata characteristics and the transformer architecture. Backed by experimentswe demonstrate that our theoretical findings are in congruence with theempirical results. We further investigate these findings in the broader contextof higher order Markov chains and deeper architectures and outline openproblems in this arena. Code is available aturlhttps://github.com/Bond1995/Markov. |


| Item |Content|
| --- |---|
|idx| 2402.04146v1 |
|title| Interpretable Multi-Source Data Fusion Through Latent Variable Gaussian Process |
|authors| Sandipp Krishnan RaviYigitcan ComlekWei ChenArjun PathakVipul GuptaRajnikant UmretiyaAndrew HoffmanGhanshyam PilaniaPiyush PanditaSayan GhoshNathaniel MckeeverLiping Wang
|links| http://arxiv.org/abs/2402.04146v1 |
|updated| 2024-02-06 16:54:59 UTC |
|summary| With the advent of artificial intelligence AI and machine learning MLvarious domains of science and engineering communites has leveraged data-drivensurrogates to model complex systems from numerous sources of informationdata. The proliferation has led to significant reduction in cost and timeinvolved in development of superior systems designed to perform specificfunctionalities. A high proposition of such surrogates are built extensivelyfusing multiple sources of data may it be published papers patents openrepositories or other resources. However not much attention has been paid tothe differences in quality and comprehensiveness of the known and unknownunderlying physical parameters of the information sources that could havedownstream implications during system optimization. Towards resolving thisissue a multi-source data fusion framework based on Latent Variable GaussianProcess LVGP is proposed. The individual data sources are tagged as acharacteristic categorical variable that are mapped into a physicallyinterpretable latent space allowing the development of source-aware datafusion modeling. Additionally a dissimilarity metric based on the latentvariables of LVGP is introduced to study and understand the differences in thesources of data. The proposed approach is demonstrated on and analyzed throughtwo mathematical representative parabola problem 2D Ackley function and twomaterials science design of FeCrAl and SmCoFe alloys case studies. From thecase studies it is observed that compared to using single-source and sourceunaware ML models the proposed multi-source data fusion framework can providebetter predictions for sparse-data problems interpretability regarding thesources and enhanced modeling capabilities by taking advantage of thecorrelations and relationships among different sources. |


| Item |Content|
| --- |---|
|idx| 2402.04114v1 |
|title| SCAFFLSA: Quantifying and Eliminating Heterogeneity Bias in Federated Linear Stochastic Approximation and Temporal Difference Learning |
|authors| Paul MangoldSergey SamsonovSafwan LabbiIlya LevinReda AlamiAlexey NaumovEric Moulines
|links| http://arxiv.org/abs/2402.04114v1 |
|updated| 2024-02-06 16:06:59 UTC |
|summary| In this paper we perform a non-asymptotic analysis of the federated linearstochastic approximation FedLSA algorithm. We explicitly quantify the biasintroduced by local training with heterogeneous agents and investigate thesample complexity of the algorithm. We show that the communication complexityof FedLSA scales polynomially with the desired precision epsilon whichlimits the benefits of federation. To overcome this we propose SCAFFLSA anovel variant of FedLSA that uses control variates to correct the bias oflocal training and prove its convergence without assumptions on statisticalheterogeneity. We apply the proposed methodology to federated temporaldifference learning with linear function approximation and analyze thecorresponding complexity improvements. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.04142v1 |
|title| Human Emotions Analysis and Recognition Using EEG Signals in Response to 360$^\circ$ Videos |
|authors| Haseeb ur Rahman AbbasiZeeshan RashidMuhammad MajidSyed Muhammad Anwar
|links| http://arxiv.org/abs/2402.04142v1 |
|updated| 2024-02-06 16:48:58 UTC |
|summary| Emotion recognition ER technology is an integral part for developinginnovative applications such as drowsiness detection and health monitoring thatplays a pivotal role in contemporary society. This study delves into ER usingelectroencephalography EEG within immersive virtual reality VRenvironments. There are four main stages in our proposed methodology includingdata acquisition pre-processing feature extraction and emotionclassification. Acknowledging the limitations of existing 2D datasets weintroduce a groundbreaking 3D VR dataset to elevate the precision of emotionelicitation. Leveraging the Interaxon Muse headband for EEG recording andOculus Quest 2 for VR stimuli we meticulously recorded data from 40participants prioritizing subjects without reported mental illnesses.Pre-processing entails rigorous cleaning uniform truncation and theapplication of a Savitzky-Golay filter to the EEG data. Feature extractionencompasses a comprehensive analysis of metrics such as power spectral densitycorrelation rational and divisional asymmetry and power spectrum. To ensurethe robustness of our model we employed a 10-fold cross-validation revealingan average validation accuracy of 85.54 with a noteworthy maximum accuracyof 90.20 in the best fold. Subsequently the trained model demonstrated acommendable test accuracy of 82.03 promising favorable outcomes. |


| Item |Content|
| --- |---|
|idx| 2402.04140v2 |
|title| Advancing Legal Reasoning: The Integration of AI to Navigate Complexities and Biases in Global Jurisprudence with Semi-Automated Arbitration Processes (SAAPs) |
|authors| Michael De'Shazer
|links| http://arxiv.org/abs/2402.04140v2 |
|updated| 2024-02-07 14:48:27 UTC |
|summary| This study consists of a novel approach toward the analysis of courtjudgments spanning five countries including the United States the UnitedKingdom Rwanda Sweden and Hong Kong. This study also explores theintersection of the latest advancements in artificial intelligence AI andlegal analysis emphasizing the role of AI specifically generative AI inidentifying human biases and facilitating automated valid and coherentmultisided argumentation of court judgments with the goal of ensuringconsistent application of laws in and across various jurisdictions. Byincorporating Advanced Language Models ALMs and a newly introduced human-AIcollaborative framework this paper seeks to analyze Grounded Theory-basedresearch design with Advanced Language Models ALMs in the practice of law.SHIRLEY is the name of the AI-based application built on top of OpenAIs GPTtechnology focusing on detecting logical inconsistencies and biases acrossvarious legal decisions. SHIRLEY analysis is aggregated and is accompanied by acomparison-oriented AI-based application called SAM also an ALM to identifyrelative deviations in SHIRLEY bias detections. Further a CRITIC is generatedwithin semi-autonomous arbitration process via the ALM SARA. A novel approachis introduced in the utilization of an AI arbitrator to critically evaluatebiases and qualitative-in-nature nuances identified by the aforementioned AIapplications SAM in concert with SHIRLEY based on the Hague Rules onBusiness and Human Rights Arbitration. This Semi-Automated Arbitration ProcessSAAP aims to uphold the integrity and fairness of legal judgments by ensuringa nuanced debate-resultant understanding through a hybrid system of AI andhuman-based collaborative analysis. |


| Item |Content|
| --- |---|
|idx| 2402.03907v1 |
|title| Embedding Large Language Models into Extended Reality: Opportunities and Challenges for Inclusion, Engagement, and Privacy |
|authors| Efe BozkirSüleyman ÖzdelKa Hei Carrie LauMengdi WangHong GaoEnkelejda Kasneci
|links| http://arxiv.org/abs/2402.03907v1 |
|updated| 2024-02-06 11:19:40 UTC |
|summary| Recent developments in computer graphics hardware artificial intelligenceAI and human-computer interaction likely lead to extended reality XRdevices and setups being more pervasive. While these devices and setups provideusers with interactive engaging and immersive experiences with differentsensing modalities such as eye and hand trackers many non-player charactersare utilized in a pre-scripted way or by conventional AI techniques. In thispaper we argue for using large language models LLMs in XR by embedding themin virtual avatars or as narratives to facilitate more inclusive experiencesthrough prompt engineering according to user profiles and fine-tuning the LLMsfor particular purposes. We argue that such inclusion will facilitate diversityfor XR use. In addition we believe that with the versatile conversationalcapabilities of LLMs users will engage more with XR environments which mighthelp XR be more used in everyday life. Lastly we speculate that combining theinformation provided to LLM-powered environments by the users and the biometricdata obtained through the sensors might lead to novel privacy invasions. Whilestudying such possible privacy invasions user privacy concerns and preferencesshould also be investigated. In summary despite some challenges embeddingLLMs into XR is a promising and novel research area with several opportunities. |


| Item |Content|
| --- |---|
|idx| 2402.03803v1 |
|title| Robot voice a voice controlled robot using arduino |
|authors| Vineeth TeedaK SujathaRakesh Mutukuru
|links| http://arxiv.org/abs/2402.03803v1 |
|updated| 2024-02-06 08:44:16 UTC |
|summary| Robotic assistants reduce the manual efforts being put in by humans in theirday-to-day tasks. In this paper we develop a voice-controlled personalassistant robot. The robot takes the human voice commands by its own built-inmicrophone. This robot not only takes the commands and executes them but alsoacknowledges them through speech output. This robot can perform differentmovements turns wakeup/shutdown operations relocate an object from one placeto another and can also develop a conversation with humans. The voice commandsare processed in real time using an offline server. The speech signal commandsare directly communicated to the server using a USB cable. The personalassistant robot is developed on a microcontroller-based platform. Performanceevaluation is carried out with encouraging results of the initial experiments.Possible improvements for applications in homes hospitals car systems andindustries are also discussed. |


| Item |Content|
| --- |---|
|idx| 2402.03750v1 |
|title| Digital Twin Mobility Profiling: A Spatio-Temporal Graph Learning Approach |
|authors| Xin ChenMingliang HouTao TangAchhardeep KaurFeng Xia
|links| http://dx.doi.org/10.1109/HPCC-DSS-SmartCity-DependSys53884.2021.00182 |
|updated| 2024-02-06 06:37:43 UTC |
|summary| With the arrival of the big data era mobility profiling has become a viablemethod of utilizing enormous amounts of mobility data to create an intelligenttransportation system. Mobility profiling can extract potential patterns inurban traffic from mobility data and is critical for a variety oftraffic-related applications. However due to the high level of complexity andthe huge amount of data mobility profiling faces huge challenges. Digital TwinDT technology paves the way for cost-effective and performance-optimisedmanagement by digitally creating a virtual representation of the network tosimulate its behaviour. In order to capture the complex spatio-temporalfeatures in traffic scenario we construct alignment diagrams to assist incompleting the spatio-temporal correlation representation and design dilatedalignment convolution network DACN to learn the fine-grained correlationsi.e. spatio-temporal interactions. We propose a digital twin mobilityprofiling DTMP framework to learn node profiles on a mobility network DTmodel. Extensive experiments have been conducted upon three real-worlddatasets. Experimental results demonstrate the effectiveness of DTMP. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.03972v1 |
|title| Joint Intrinsic Motivation for Coordinated Exploration in Multi-Agent Deep Reinforcement Learning |
|authors| Maxime ToquebiauNicolas BredecheFaïz BenamarJae-Yun Jun
|links| http://arxiv.org/abs/2402.03972v1 |
|updated| 2024-02-06 13:02:00 UTC |
|summary| Multi-agent deep reinforcement learning MADRL problems often encounter thechallenge of sparse rewards. This challenge becomes even more pronounced whencoordination among agents is necessary. As performance depends not only on oneagents behavior but rather on the joint behavior of multiple agents findingan adequate solution becomes significantly harder. In this context a group ofagents can benefit from actively exploring different joint strategies in orderto determine the most efficient one. In this paper we propose an approach forrewarding strategies where agents collectively exhibit novel behaviors. Wepresent JIM Joint Intrinsic Motivation a multi-agent intrinsic motivationmethod that follows the centralized learning with decentralized executionparadigm. JIM rewards joint trajectories based on a centralized measure ofnovelty designed to function in continuous environments. We demonstrate thestrengths of this approach both in a synthetic environment designed to revealshortcomings of state-of-the-art MADRL methods and in simulated robotic tasks.Results show that joint exploration is crucial for solving tasks where theoptimal strategy requires a high level of coordination. |


| Item |Content|
| --- |---|
|idx| 2402.03928v1 |
|title| Approximating the Core via Iterative Coalition Sampling |
|authors| Ian GempMarc LanctotLuke MarrisYiran MaoEdgar Duéñez-GuzmánSarah PerrinAndras GyorgyRomuald ElieGeorgios PiliourasMichael KaisersDaniel HennesKalesha BullardKate LarsonYoram Bachrach
|links| http://arxiv.org/abs/2402.03928v1 |
|updated| 2024-02-06 11:54:48 UTC |
|summary| The core is a central solution concept in cooperative game theory defined asthe set of feasible allocations or payments such that no subset of agents hasincentive to break away and form their own subgroup or coalition. However ithas long been known that the core and approximations such as the least-coreare hard to compute. This limits our ability to analyze cooperative games ingeneral and to fully embrace cooperative game theory contributions in domainssuch as explainable AI XAI where the core can complement the Shapley valuesto identify influential features or instances supporting predictions byblack-box models. We propose novel iterative algorithms for computing variantsof the core which avoid the computational bottleneck of many other approachesnamely solving large linear programs. As such they scale better to very largeproblems as we demonstrate across different classes of cooperative gamesincluding weighted voting games induced subgraph games and marginalcontribution networks. We also explore our algorithms in the context of XAIproviding further evidence of the power of the core for such applications. |


| Item |Content|
| --- |---|
|idx| 2402.03669v1 |
|title| Convergence Analysis of Distributed Generalized Nash Equilibria Seeking Algorithm with Asynchrony and Delays |
|authors| Huaqing LiLiang RanLifeng ZhengZhe LiJinhui HuJun LiTingwen Huang
|links| http://arxiv.org/abs/2402.03669v1 |
|updated| 2024-02-06 03:46:26 UTC |
|summary| This paper considers a class of noncooperative games in which the feasibledecision sets of all players are coupled together by a coupled inequalityconstraint. Adopting the variational inequality formulation of the game wefirst introduce a new local edge-based equilibrium condition and develop adistributed primal-dual proximal algorithm with full information. Consideringchallenges when communication delays occur we devise an asynchronousdistributed algorithm to seek a generalized Nash equilibrium. This asynchronousscheme arbitrarily activates one player to start new computations independentlyat different iteration instants which means that the picked player can use theinvolved out-dated information from itself and its neighbors to perform newupdates. A distinctive attribute is that the proposed algorithms enable thederivation of new distributed forward-backward-like extensions. In theoreticalaspect we provide explicit conditions on algorithm parameters for instancethe step-sizes to establish a sublinear convergence rate for the proposedsynchronous algorithm. Moreover the asynchronous algorithm guarantees almostsure convergence in expectation under the same step-size conditions and somestandard assumptions. An interesting observation is that our analysis approachimproves the convergence rate of prior synchronous distributedforward-backward-based algorithms. Finally the viability and performance ofthe proposed algorithms are demonstrated by numerical studies on the networkedCournot competition. |


| Item |Content|
| --- |---|
|idx| 2402.03653v1 |
|title| Agent-Based Triangle Counting and its Applications in Anonymous Graphs |
|authors| Prabhat Kumar ChandApurba DasAnisur Rahaman Molla
|links| http://arxiv.org/abs/2402.03653v1 |
|updated| 2024-02-06 03:00:12 UTC |
|summary| Triangle counting in a graph is a fundamental problem and has a wide range ofapplications in various domains. It is crucial in understanding the structuralproperties of a graph and is often used as a building block for more complexgraph analytics. In this paper we solve the triangle counting problem in ananonymous graph in a distributed setting using mobile agents and subsequentlyuse this as a subroutine to tackle the truss decomposition and trianglecentrality problem. The paper employs mobile agents placed on the nodes of thegraph to coordinate among themselves to solve the triangle enumeration problemfor the graph. Following the literature we consider the synchronous systemswhere each robot executes its tasks concurrently with all others and hence timecomplexity can be measured as the number of rounds needed to complete the task.The graph is anonymous i.e. without any node labels or IDs but the agentsare autonomous with distinct IDs and have limited memory. Agents can onlycommunicate with other agents locally i.e. if and only if they are at the samenode. The goal is to devise algorithms that minimise both the time required fortriangle counting and the memory usage at each agent. We further demonstratehow the triangle count obtained through the mobile agent approach can beleveraged to address the truss decomposition triangle centrality and localclustering coefficient problems which involves finding maximal sub-graphs withstrong interconnections. Truss decomposition helps in identifying maximalhighly interconnected sub-graphs or trusses within a network thus revealingthe structural cohesion and tight-knit communities in complex graphsfacilitating the analysis of relationships and information flow in variousfields such as social networks biology and recommendation systems. |


| Item |Content|
| --- |---|
|idx| 2402.03590v1 |
|title| Assessing the Impact of Distribution Shift on Reinforcement Learning Performance |
|authors| Ted FujimotoJoshua SuetterleinSamrat ChatterjeeAuroop Ganguly
|links| http://arxiv.org/abs/2402.03590v1 |
|updated| 2024-02-05 23:50:55 UTC |
|summary| Research in machine learning is making progress in fixing its ownreproducibility crisis. Reinforcement learning RL in particular faces itsown set of unique challenges. Comparison of point estimates and plots thatshow successful convergence to the optimal policy during training mayobfuscate overfitting or dependence on the experimental setup. Althoughresearchers in RL have proposed reliability metrics that account foruncertainty to better understand each algorithms strengths and weaknesses therecommendations of past work do not assume the presence of out-of-distributionobservations. We propose a set of evaluation methods that measure therobustness of RL algorithms under distribution shifts. The tools presented hereargue for the need to account for performance over time while the agent isacting in its environment. In particular we recommend time series analysis asa method of observational RL evaluation. We also show that the uniqueproperties of RL and simulated dynamic environments allow us to make strongerassumptions to justify the measurement of causal impact in our evaluations. Wethen apply these tools to single-agent and multi-agent environments to show theimpact of introducing distribution shifts during test time. We present thismethodology as a first step toward rigorous RL evaluation in the presence ofdistribution shifts. |


