# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.21075v1 |
|title| Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis |
|authors| Chaoyou FuYuhan DaiYondong LuoLei LiShuhuai RenRenrui ZhangZihan WangChenyu ZhouYunhang ShenMengdan ZhangPeixian ChenYanwei LiShaohui LinSirui ZhaoKe LiTong XuXiawu ZhengEnhong ChenRongrong JiXing Sun
|links| http://arxiv.org/abs/2405.21075v1 |
|updated| 2024-05-31 17:59:47 UTC |
|summary| In the quest for artificial general intelligence Multi-modal Large LanguageModels MLLMs have emerged as a focal point in recent advancements. Howeverthe predominant focus remains on developing their capabilities in static imageunderstanding. The potential of MLLMs in processing sequential visual data isstill insufficiently explored highlighting the absence of a comprehensivehigh-quality assessment of their performance. In this paper we introduceVideo-MME the first-ever full-spectrum Multi-Modal Evaluation benchmark ofMLLMs in Video analysis. Our work distinguishes from existing benchmarksthrough four key features: 1 Diversity in video types spanning 6 primaryvisual domains with 30 subfields to ensure broad scenario generalizability 2Duration in temporal dimension encompassing both short- medium- andlong-term videos ranging from 11 seconds to 1 hour for robust contextualdynamics 3 Breadth in data modalities integrating multi-modal inputs besidesvideo frames including subtitles and audios to unveil the all-roundcapabilities of MLLMs 4 Quality in annotations utilizing rigorous manuallabeling by expert annotators to facilitate precise and reliable modelassessment. 900 videos with a total of 256 hours are manually selected andannotated by repeatedly viewing all the video content resulting in 2700question-answer pairs. With Video-MME we extensively evaluate variousstate-of-the-art MLLMs including GPT-4 series and Gemini 1.5 Pro as well asopen-source image models like InternVL-Chat-V1.5 and video models likeLLaVA-NeXT-Video. Our experiments reveal that Gemini 1.5 Pro is thebest-performing commercial model significantly outperforming the open-sourcemodels. Our dataset along with these findings underscores the need for furtherimprovements in handling longer sequences and multi-modal data. Project Page:https://video-mme.github.io |


| Item |Content|
| --- |---|
|idx| 2405.21070v1 |
|title| Generalization Beyond Data Imbalance: A Controlled Study on CLIP for Transferable Insights |
|authors| Xin WenBingchen ZhaoYilun ChenJiangmiao PangXiaojuan Qi
|links| http://arxiv.org/abs/2405.21070v1 |
|updated| 2024-05-31 17:57:24 UTC |
|summary| Severe data imbalance naturally exists among web-scale vision-languagedatasets. Despite this we find CLIP pre-trained thereupon exhibits notablerobustness to the data imbalance compared to supervised learning anddemonstrates significant effectiveness in learning generalizablerepresentations. With an aim to investigate the reasons behind this finding weconduct controlled experiments to study various underlying factors and revealthat CLIPs pretext task forms a dynamic classification problem wherein only asubset of classes is present in training. This isolates the bias from dominantclasses and implicitly balances the learning signal. Furthermore therobustness and discriminability of CLIP improve with more descriptive languagesupervision larger data scale and broader open-world concepts which areinaccessible to supervised learning. Our study not only uncovers the mechanismsbehind CLIPs generalizability beyond data imbalance but also providestransferable insights for the research community. The findings are validated inboth supervised and self-supervised learning enabling models trained onimbalanced data to achieve CLIP-level performance on diverse recognition tasks.Code will be available at: https://github.com/CVMI-Lab/clip-beyond-tail. |


| Item |Content|
| --- |---|
|idx| 2405.21068v1 |
|title| Code Pretraining Improves Entity Tracking Abilities of Language Models |
|authors| Najoung KimSebastian SchusterShubham Toshniwal
|links| http://arxiv.org/abs/2405.21068v1 |
|updated| 2024-05-31 17:56:33 UTC |
|summary| Recent work has provided indirect evidence that pretraining language modelson code improves the ability of models to track state changes of discourseentities expressed in natural language. In this work we systematically testthis claim by comparing pairs of language models on their entity trackingperformance. Critically the pairs consist of base models and models trained ontop of these base models with additional code data. We extend this analysis toadditionally examine the effect of math training another highly structureddata type and alignment tuning an important step for enhancing the usabilityof models. We find clear evidence that models additionally trained on largeamounts of code outperform the base models. On the other hand we find noconsistent benefit of additional math training or alignment tuning acrossvarious model families. |


| Item |Content|
| --- |---|
|idx| 2405.21047v1 |
|title| Grammar-Aligned Decoding |
|authors| Kanghee ParkJiayu WangTaylor Berg-KirkpatrickNadia PolikarpovaLoris D'Antoni
|links| http://arxiv.org/abs/2405.21047v1 |
|updated| 2024-05-31 17:39:15 UTC |
|summary| Large Language Models LLMs struggle with reliably generating highlystructured outputs such as program code mathematical formulas or well-formedmarkup. Constrained decoding approaches mitigate this problem by greedilyrestricting what tokens an LLM can output at each step to guarantee that theoutput matches a given constraint. Specifically in grammar-constraineddecoding GCD the LLMs output must follow a given grammar. In this paper wedemonstrate that GCD techniques and in general constrained decodingtechniques can distort the LLMs distribution leading to outputs that aregrammatical but appear with likelihoods that are not proportional to the onesgiven by the LLM and so ultimately are low-quality. We call the problem ofaligning sampling with a grammar constraint grammar-aligned decoding GADand propose adaptive sampling with approximate expected futures ASAp adecoding algorithm that guarantees the output to be grammatical while provablyproducing outputs that match the conditional probability of the LLMsdistribution conditioned on the given grammar constraint. Our algorithm usesprior sample outputs to soundly overapproximate the future grammaticality ofdifferent output prefixes. Our evaluation on code generation and structured NLPtasks shows how ASAp often produces outputs with higher likelihood accordingto the LLMs distribution than existing GCD techniques while still enforcingthe desired grammatical constraints. |


| Item |Content|
| --- |---|
|idx| 2405.21046v1 |
|title| Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF |
|authors| Tengyang XieDylan J. FosterAkshay KrishnamurthyCorby RossetAhmed AwadallahAlexander Rakhlin
|links| http://arxiv.org/abs/2405.21046v1 |
|updated| 2024-05-31 17:39:06 UTC |
|summary| Reinforcement learning from human feedback RLHF has emerged as a centraltool for language model alignment. We consider online exploration in RLHFwhich exploits interactive access to human or AI feedback by deliberatelyencouraging the model to produce diverse maximally informative responses. Byallowing RLHF to confidently stray from the pre-trained model onlineexploration offers the possibility of novel potentially super-humancapabilities but its full potential as a paradigm for language model traininghas yet to be realized owing to computational and statistical bottlenecks indirectly adapting existing reinforcement learning techniques. We propose a newalgorithm for online exploration in RLHF Exploratory Preference OptimizationXPO which is simple and practical -- a one-line change to online DirectPreference Optimization DPO Rafailov et al. 2023 -- yet enjoys thestrongest known provable guarantees and promising empirical performance. XPOaugments the DPO objective with a novel and principled exploration bonusempowering the algorithm to explore outside the support of the initial modeland human feedback data. In theory we show that XPO is provablysample-efficient and converges to a near-optimal language model policy undernatural exploration conditions irrespective of whether the initial model hasgood coverage. Our analysis which builds on the observation that DPOimplicitly performs a form of Qstar-approximation or Bellman errorminimization combines previously disparate techniques from language modelingand theoretical reinforcement learning in a serendipitous fashion through theperspective of KL-regularized Markov decision processes. Empirically we findthat XPO is more sample-efficient than non-exploratory DPO variants in apreliminary evaluation. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.21068v1 |
|title| Code Pretraining Improves Entity Tracking Abilities of Language Models |
|authors| Najoung KimSebastian SchusterShubham Toshniwal
|links| http://arxiv.org/abs/2405.21068v1 |
|updated| 2024-05-31 17:56:33 UTC |
|summary| Recent work has provided indirect evidence that pretraining language modelson code improves the ability of models to track state changes of discourseentities expressed in natural language. In this work we systematically testthis claim by comparing pairs of language models on their entity trackingperformance. Critically the pairs consist of base models and models trained ontop of these base models with additional code data. We extend this analysis toadditionally examine the effect of math training another highly structureddata type and alignment tuning an important step for enhancing the usabilityof models. We find clear evidence that models additionally trained on largeamounts of code outperform the base models. On the other hand we find noconsistent benefit of additional math training or alignment tuning acrossvarious model families. |


| Item |Content|
| --- |---|
|idx| 2405.21064v1 |
|title| Recurrent neural networks: vanishing and exploding gradients are not the end of the story |
|authors| Nicolas ZucchetAntonio Orvieto
|links| http://arxiv.org/abs/2405.21064v1 |
|updated| 2024-05-31 17:53:00 UTC |
|summary| Recurrent neural networks RNNs notoriously struggle to learn long-termmemories primarily due to vanishing and exploding gradients. The recentsuccess of state-space models SSMs a subclass of RNNs to overcome suchdifficulties challenges our theoretical understanding. In this paper we delveinto the optimization challenges of RNNs and discover that as the memory of anetwork increases changes in its parameters result in increasingly largeoutput variations making gradient-based learning highly sensitive evenwithout exploding gradients. Our analysis further reveals the importance of theelement-wise recurrence design pattern combined with careful parametrizationsin mitigating this effect. This feature is present in SSMs as well as in otherarchitectures such as LSTMs. Overall our insights provide a new explanationfor some of the difficulties in gradient-based learning of RNNs and why somearchitectures perform better than others. |


| Item |Content|
| --- |---|
|idx| 2405.21063v1 |
|title| Neural Network Verification with Branch-and-Bound for General Nonlinearities |
|authors| Zhouxing ShiQirui JinZico KolterSuman JanaCho-Jui HsiehHuan Zhang
|links| http://arxiv.org/abs/2405.21063v1 |
|updated| 2024-05-31 17:51:07 UTC |
|summary| Branch-and-bound BaB is among the most effective methods for neural networkNN verification. However existing works on BaB have mostly focused on NNswith piecewise linear activations especially ReLU networks. In this paper wedevelop a general framework named GenBaB to conduct BaB for generalnonlinearities in general computational graphs based on linear boundpropagation. To decide which neuron to branch we design a new branchingheuristic which leverages linear bounds as shortcuts to efficiently estimatethe potential improvement after branching. To decide nontrivial branchingpoints for general nonlinear functions we propose to optimize branching pointsoffline which can be efficiently leveraged during verification with a lookuptable. We demonstrate the effectiveness of our GenBaB on verifying a wide rangeof NNs including networks with activation functions such as Sigmoid TanhSine and GeLU as well as networks involving multi-dimensional nonlinearoperations such as multiplications in LSTMs and Vision Transformers. Ourframework also allows the verification of general nonlinear computation graphsand enables verification applications beyond simple neural networksparticularly for AC Optimal Power Flow ACOPF. GenBaB is part of the latestalphabeta-CROWN the winner of the 4th International Verification ofNeural Networks Competition VNN-COMP 2023. |


| Item |Content|
| --- |---|
|idx| 2405.21056v1 |
|title| An Organic Weed Control Prototype using Directed Energy and Deep Learning |
|authors| Deng CaoHongbo ZhangRajveer Dhillon
|links| http://arxiv.org/abs/2405.21056v1 |
|updated| 2024-05-31 17:47:22 UTC |
|summary| Organic weed control is a vital to improve crop yield with a sustainableapproach. In this work a directed energy weed control robot prototypespecifically designed for organic farms is proposed. The robot uses a noveldistributed array robot DAR unit for weed treatment. Soybean and corndatabases are built to train deep learning neural nets to perform weedrecognition. The initial deep learning neural nets show a high performance inclassifying crops. The robot uses a patented directed energy plant eradicationrecipe that is completely organic and UV-C free with no chemical damage orphysical disturbance to the soil. The deep learning can classify 8 common weedspecies in a soybean field under natural environment with up to 98 accuracy. |


| Item |Content|
| --- |---|
|idx| 2405.21047v1 |
|title| Grammar-Aligned Decoding |
|authors| Kanghee ParkJiayu WangTaylor Berg-KirkpatrickNadia PolikarpovaLoris D'Antoni
|links| http://arxiv.org/abs/2405.21047v1 |
|updated| 2024-05-31 17:39:15 UTC |
|summary| Large Language Models LLMs struggle with reliably generating highlystructured outputs such as program code mathematical formulas or well-formedmarkup. Constrained decoding approaches mitigate this problem by greedilyrestricting what tokens an LLM can output at each step to guarantee that theoutput matches a given constraint. Specifically in grammar-constraineddecoding GCD the LLMs output must follow a given grammar. In this paper wedemonstrate that GCD techniques and in general constrained decodingtechniques can distort the LLMs distribution leading to outputs that aregrammatical but appear with likelihoods that are not proportional to the onesgiven by the LLM and so ultimately are low-quality. We call the problem ofaligning sampling with a grammar constraint grammar-aligned decoding GADand propose adaptive sampling with approximate expected futures ASAp adecoding algorithm that guarantees the output to be grammatical while provablyproducing outputs that match the conditional probability of the LLMsdistribution conditioned on the given grammar constraint. Our algorithm usesprior sample outputs to soundly overapproximate the future grammaticality ofdifferent output prefixes. Our evaluation on code generation and structured NLPtasks shows how ASAp often produces outputs with higher likelihood accordingto the LLMs distribution than existing GCD techniques while still enforcingthe desired grammatical constraints. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.21070v1 |
|title| Generalization Beyond Data Imbalance: A Controlled Study on CLIP for Transferable Insights |
|authors| Xin WenBingchen ZhaoYilun ChenJiangmiao PangXiaojuan Qi
|links| http://arxiv.org/abs/2405.21070v1 |
|updated| 2024-05-31 17:57:24 UTC |
|summary| Severe data imbalance naturally exists among web-scale vision-languagedatasets. Despite this we find CLIP pre-trained thereupon exhibits notablerobustness to the data imbalance compared to supervised learning anddemonstrates significant effectiveness in learning generalizablerepresentations. With an aim to investigate the reasons behind this finding weconduct controlled experiments to study various underlying factors and revealthat CLIPs pretext task forms a dynamic classification problem wherein only asubset of classes is present in training. This isolates the bias from dominantclasses and implicitly balances the learning signal. Furthermore therobustness and discriminability of CLIP improve with more descriptive languagesupervision larger data scale and broader open-world concepts which areinaccessible to supervised learning. Our study not only uncovers the mechanismsbehind CLIPs generalizability beyond data imbalance but also providestransferable insights for the research community. The findings are validated inboth supervised and self-supervised learning enabling models trained onimbalanced data to achieve CLIP-level performance on diverse recognition tasks.Code will be available at: https://github.com/CVMI-Lab/clip-beyond-tail. |


| Item |Content|
| --- |---|
|idx| 2405.21064v1 |
|title| Recurrent neural networks: vanishing and exploding gradients are not the end of the story |
|authors| Nicolas ZucchetAntonio Orvieto
|links| http://arxiv.org/abs/2405.21064v1 |
|updated| 2024-05-31 17:53:00 UTC |
|summary| Recurrent neural networks RNNs notoriously struggle to learn long-termmemories primarily due to vanishing and exploding gradients. The recentsuccess of state-space models SSMs a subclass of RNNs to overcome suchdifficulties challenges our theoretical understanding. In this paper we delveinto the optimization challenges of RNNs and discover that as the memory of anetwork increases changes in its parameters result in increasingly largeoutput variations making gradient-based learning highly sensitive evenwithout exploding gradients. Our analysis further reveals the importance of theelement-wise recurrence design pattern combined with careful parametrizationsin mitigating this effect. This feature is present in SSMs as well as in otherarchitectures such as LSTMs. Overall our insights provide a new explanationfor some of the difficulties in gradient-based learning of RNNs and why somearchitectures perform better than others. |


| Item |Content|
| --- |---|
|idx| 2405.21063v1 |
|title| Neural Network Verification with Branch-and-Bound for General Nonlinearities |
|authors| Zhouxing ShiQirui JinZico KolterSuman JanaCho-Jui HsiehHuan Zhang
|links| http://arxiv.org/abs/2405.21063v1 |
|updated| 2024-05-31 17:51:07 UTC |
|summary| Branch-and-bound BaB is among the most effective methods for neural networkNN verification. However existing works on BaB have mostly focused on NNswith piecewise linear activations especially ReLU networks. In this paper wedevelop a general framework named GenBaB to conduct BaB for generalnonlinearities in general computational graphs based on linear boundpropagation. To decide which neuron to branch we design a new branchingheuristic which leverages linear bounds as shortcuts to efficiently estimatethe potential improvement after branching. To decide nontrivial branchingpoints for general nonlinear functions we propose to optimize branching pointsoffline which can be efficiently leveraged during verification with a lookuptable. We demonstrate the effectiveness of our GenBaB on verifying a wide rangeof NNs including networks with activation functions such as Sigmoid TanhSine and GeLU as well as networks involving multi-dimensional nonlinearoperations such as multiplications in LSTMs and Vision Transformers. Ourframework also allows the verification of general nonlinear computation graphsand enables verification applications beyond simple neural networksparticularly for AC Optimal Power Flow ACOPF. GenBaB is part of the latestalphabeta-CROWN the winner of the 4th International Verification ofNeural Networks Competition VNN-COMP 2023. |


| Item |Content|
| --- |---|
|idx| 2405.21061v1 |
|title| Graph External Attention Enhanced Transformer |
|authors| Jianqing LiangMin ChenJiye Liang
|links| http://arxiv.org/abs/2405.21061v1 |
|updated| 2024-05-31 17:50:27 UTC |
|summary| The Transformer architecture has recently gained considerable attention inthe field of graph representation learning as it naturally overcomes severallimitations of Graph Neural Networks GNNs with customized attentionmechanisms or positional and structural encodings. Despite making someprogress existing works tend to overlook external information of graphsspecifically the correlation between graphs. Intuitively graphs with similarstructures should have similar representations. Therefore we propose GraphExternal Attention GEA -- a novel attention mechanism that leverages multipleexternal node/edge key-value units to capture inter-graph correlationsimplicitly. On this basis we design an effective architecture called GraphExternal Attention Enhanced Transformer GEAET which integrates localstructure and global interaction information for more comprehensive graphrepresentations. Extensive experiments on benchmark datasets demonstrate thatGEAET achieves state-of-the-art empirical performance. The source code isavailable for reproducibility at: https://github.com/icm1018/GEAET. |


| Item |Content|
| --- |---|
|idx| 2405.21060v1 |
|title| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |
|authors| Tri DaoAlbert Gu
|links| http://arxiv.org/abs/2405.21060v1 |
|updated| 2024-05-31 17:50:01 UTC |
|summary| While Transformers have been the main architecture behind deep learningssuccess in language modeling state-space models SSMs such as Mamba haverecently been shown to match or outperform Transformers at small to mediumscale. We show that these families of models are actually quite closelyrelated and develop a rich framework of theoretical connections between SSMsand variants of attention connected through various decompositions of awell-studied class of structured semiseparable matrices. Our state spaceduality SSD framework allows us to design a new architecture Mamba-2 whosecore layer is an a refinement of Mambas selective SSM that is 2-8X fasterwhile continuing to be competitive with Transformers on language modeling. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.21075v1 |
|title| Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis |
|authors| Chaoyou FuYuhan DaiYondong LuoLei LiShuhuai RenRenrui ZhangZihan WangChenyu ZhouYunhang ShenMengdan ZhangPeixian ChenYanwei LiShaohui LinSirui ZhaoKe LiTong XuXiawu ZhengEnhong ChenRongrong JiXing Sun
|links| http://arxiv.org/abs/2405.21075v1 |
|updated| 2024-05-31 17:59:47 UTC |
|summary| In the quest for artificial general intelligence Multi-modal Large LanguageModels MLLMs have emerged as a focal point in recent advancements. Howeverthe predominant focus remains on developing their capabilities in static imageunderstanding. The potential of MLLMs in processing sequential visual data isstill insufficiently explored highlighting the absence of a comprehensivehigh-quality assessment of their performance. In this paper we introduceVideo-MME the first-ever full-spectrum Multi-Modal Evaluation benchmark ofMLLMs in Video analysis. Our work distinguishes from existing benchmarksthrough four key features: 1 Diversity in video types spanning 6 primaryvisual domains with 30 subfields to ensure broad scenario generalizability 2Duration in temporal dimension encompassing both short- medium- andlong-term videos ranging from 11 seconds to 1 hour for robust contextualdynamics 3 Breadth in data modalities integrating multi-modal inputs besidesvideo frames including subtitles and audios to unveil the all-roundcapabilities of MLLMs 4 Quality in annotations utilizing rigorous manuallabeling by expert annotators to facilitate precise and reliable modelassessment. 900 videos with a total of 256 hours are manually selected andannotated by repeatedly viewing all the video content resulting in 2700question-answer pairs. With Video-MME we extensively evaluate variousstate-of-the-art MLLMs including GPT-4 series and Gemini 1.5 Pro as well asopen-source image models like InternVL-Chat-V1.5 and video models likeLLaVA-NeXT-Video. Our experiments reveal that Gemini 1.5 Pro is thebest-performing commercial model significantly outperforming the open-sourcemodels. Our dataset along with these findings underscores the need for furtherimprovements in handling longer sequences and multi-modal data. Project Page:https://video-mme.github.io |


| Item |Content|
| --- |---|
|idx| 2405.21074v1 |
|title| Latent Intrinsics Emerge from Training to Relight |
|authors| Xiao ZhangWilliam GaoSeemandhar JainMichael MaireDavid. A. ForsythAnand Bhattad
|links| http://arxiv.org/abs/2405.21074v1 |
|updated| 2024-05-31 17:59:12 UTC |
|summary| Image relighting is the task of showing what a scene from a source imagewould look like if illuminated differently. Inverse graphics schemes recover anexplicit representation of geometry and a set of chosen intrinsics thenrelight with some form of renderer. However error control for inverse graphicsis difficult and inverse graphics methods can represent only the effects ofthe chosen intrinsics. This paper describes a relighting method that isentirely data-driven where intrinsics and lighting are each represented aslatent variables. Our approach produces SOTA relightings of real scenes asmeasured by standard metrics. We show that albedo can be recovered from ourlatent intrinsics without using any example albedos and that the albedosrecovered are competitive with SOTA methods. |


| Item |Content|
| --- |---|
|idx| 2405.21070v1 |
|title| Generalization Beyond Data Imbalance: A Controlled Study on CLIP for Transferable Insights |
|authors| Xin WenBingchen ZhaoYilun ChenJiangmiao PangXiaojuan Qi
|links| http://arxiv.org/abs/2405.21070v1 |
|updated| 2024-05-31 17:57:24 UTC |
|summary| Severe data imbalance naturally exists among web-scale vision-languagedatasets. Despite this we find CLIP pre-trained thereupon exhibits notablerobustness to the data imbalance compared to supervised learning anddemonstrates significant effectiveness in learning generalizablerepresentations. With an aim to investigate the reasons behind this finding weconduct controlled experiments to study various underlying factors and revealthat CLIPs pretext task forms a dynamic classification problem wherein only asubset of classes is present in training. This isolates the bias from dominantclasses and implicitly balances the learning signal. Furthermore therobustness and discriminability of CLIP improve with more descriptive languagesupervision larger data scale and broader open-world concepts which areinaccessible to supervised learning. Our study not only uncovers the mechanismsbehind CLIPs generalizability beyond data imbalance but also providestransferable insights for the research community. The findings are validated inboth supervised and self-supervised learning enabling models trained onimbalanced data to achieve CLIP-level performance on diverse recognition tasks.Code will be available at: https://github.com/CVMI-Lab/clip-beyond-tail. |


| Item |Content|
| --- |---|
|idx| 2405.21066v1 |
|title| Mixed Diffusion for 3D Indoor Scene Synthesis |
|authors| Siyi HuDiego Martin ArroyoStephanie DebatsFabian ManhardtLuca CarloneFederico Tombari
|links| http://arxiv.org/abs/2405.21066v1 |
|updated| 2024-05-31 17:54:52 UTC |
|summary| Realistic conditional 3D scene synthesis significantly enhances andaccelerates the creation of virtual environments which can also provideextensive training data for computer vision and robotics research among otherapplications. Diffusion models have shown great performance in relatedapplications e.g. making precise arrangements of unordered sets. Howeverthese models have not been fully explored in floor-conditioned scene synthesisproblems. We present MiDiffusion a novel mixed discrete-continuous diffusionmodel architecture designed to synthesize plausible 3D indoor scenes fromgiven room types floor plans and potentially pre-existing objects. Werepresent a scene layout by a 2D floor plan and a set of objects each definedby its category location size and orientation. Our approach uniquelyimplements structured corruption across the mixed discrete semantic andcontinuous geometric domains resulting in a better conditioned problem for thereverse denoising step. We evaluate our approach on the 3D-FRONT dataset. Ourexperimental results demonstrate that MiDiffusion substantially outperformsstate-of-the-art autoregressive and diffusion models in floor-conditioned 3Dscene synthesis. In addition our models can handle partial object constraintsvia a corruption-and-masking strategy without task specific training. We showMiDiffusion maintains clear advantages over existing approaches in scenecompletion and furniture arrangement experiments. |


| Item |Content|
| --- |---|
|idx| 2405.21059v1 |
|title| Unified Directly Denoising for Both Variance Preserving and Variance Exploding Diffusion Models |
|authors| Jingjing WangDan ZhangFeng Luo
|links| http://arxiv.org/abs/2405.21059v1 |
|updated| 2024-05-31 17:49:51 UTC |
|summary| Previous work has demonstrated that in the Variance Preserving VPscenario the nascent Directly Denoising Diffusion Models DDDM can generatehigh-quality images in one step while achieving even better performance inmultistep sampling. However the Pseudo-LPIPS loss used in DDDM leads toconcerns about the bias in assessment. Here we propose a unified DDDM uDDDMframework that generates images in one-step/multiple steps for both VariancePreserving VP and Variance Exploding VE cases. We provide theoreticalproofs of the existence and uniqueness of the models solution paths as wellas the non-intersecting property of the sampling paths. Additionally wepropose an adaptive Pseudo-Huber loss function to balance the convergence tothe true solution and the stability of convergence process.Through acomprehensive evaluation we demonstrate that uDDDMs achieve FID scorescomparable to the best-performing methods available for CIFAR-10 in both VP andVE. Specifically uDDDM achieves one-step generation on CIFAR10 with FID of2.63 and 2.53 for VE and VP respectively. By extending the sampling to 1000steps we further reduce FID score to 1.71 and 1.65 for VE and VP respectivelysetting state-of-the-art performance in both cases. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.21046v1 |
|title| Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF |
|authors| Tengyang XieDylan J. FosterAkshay KrishnamurthyCorby RossetAhmed AwadallahAlexander Rakhlin
|links| http://arxiv.org/abs/2405.21046v1 |
|updated| 2024-05-31 17:39:06 UTC |
|summary| Reinforcement learning from human feedback RLHF has emerged as a centraltool for language model alignment. We consider online exploration in RLHFwhich exploits interactive access to human or AI feedback by deliberatelyencouraging the model to produce diverse maximally informative responses. Byallowing RLHF to confidently stray from the pre-trained model onlineexploration offers the possibility of novel potentially super-humancapabilities but its full potential as a paradigm for language model traininghas yet to be realized owing to computational and statistical bottlenecks indirectly adapting existing reinforcement learning techniques. We propose a newalgorithm for online exploration in RLHF Exploratory Preference OptimizationXPO which is simple and practical -- a one-line change to online DirectPreference Optimization DPO Rafailov et al. 2023 -- yet enjoys thestrongest known provable guarantees and promising empirical performance. XPOaugments the DPO objective with a novel and principled exploration bonusempowering the algorithm to explore outside the support of the initial modeland human feedback data. In theory we show that XPO is provablysample-efficient and converges to a near-optimal language model policy undernatural exploration conditions irrespective of whether the initial model hasgood coverage. Our analysis which builds on the observation that DPOimplicitly performs a form of Qstar-approximation or Bellman errorminimization combines previously disparate techniques from language modelingand theoretical reinforcement learning in a serendipitous fashion through theperspective of KL-regularized Markov decision processes. Empirically we findthat XPO is more sample-efficient than non-exploratory DPO variants in apreliminary evaluation. |


| Item |Content|
| --- |---|
|idx| 2405.21037v1 |
|title| Introducing sgboost: A Practical Guide and Implementation of sparse-group boosting in R |
|authors| Fabian ObsterChristian Heumann
|links| http://arxiv.org/abs/2405.21037v1 |
|updated| 2024-05-31 17:29:51 UTC |
|summary| This paper introduces the sgboost package in R which implements sparse-groupboosting for modeling high-dimensional data with natural groupings incovariates. Sparse-group boosting offers a flexible approach for both group andindividual variable selection reducing overfitting and enhancing modelinterpretability. The package uses regularization techniques based on thedegrees of freedom of individual and group base-learners and is designed to beused in conjunction with the mboost package. Through comparisons with existingmethods and demonstration of its unique functionalities this paper provides apractical guide on utilizing sparse-group boosting in R accompanied by codeexamples to facilitate its application in various research domains. Overallthis paper serves as a valuable resource for researchers and practitionersseeking to use sparse-group boosting for efficient and interpretablehigh-dimensional data analysis. |


| Item |Content|
| --- |---|
|idx| 2405.20970v1 |
|title| PUAL: A Classifier on Trifurcate Positive-Unlabeled Data |
|authors| Xiaoke WangXiaochen YangRui ZhuJing-Hao Xue
|links| http://arxiv.org/abs/2405.20970v1 |
|updated| 2024-05-31 16:18:06 UTC |
|summary| Positive-unlabeled PU learning aims to train a classifier using the datacontaining only labeled-positive instances and unlabeled instances. Howeverexisting PU learning methods are generally hard to achieve satisfactoryperformance on trifurcate data where the positive instances distribute on bothsides of the negative instances. To address this issue firstly we propose a PUclassifier with asymmetric loss PUAL by introducing a structure ofasymmetric loss on positive instances into the objective function of the globaland local learning classifier. Then we develop a kernel-based algorithm toenable PUAL to obtain non-linear decision boundary. We show that throughexperiments on both simulated and real-world datasets PUAL can achievesatisfactory classification on trifurcate data. |


| Item |Content|
| --- |---|
|idx| 2405.20954v1 |
|title| Aligning Multiclass Neural Network Classifier Criterion with Task Performance via $F_β$-Score |
|authors| Nathan TsoiDeyuan LiTaesoo Daniel LeeMarynel Vázquez
|links| http://arxiv.org/abs/2405.20954v1 |
|updated| 2024-05-31 15:54:01 UTC |
|summary| Multiclass neural network classifiers are typically trained usingcross-entropy loss. Following training the performance of this same neuralnetwork is evaluated using an application-specific metric based on themulticlass confusion matrix such as the Macro F_beta-Score. It isquestionable whether the use of cross-entropy will yield a classifier thataligns with the intended application-specific performance criteriaparticularly in scenarios where there is a need to emphasize one aspect ofclassifier performance. For example if greater precision is preferred overrecall the beta value in the F_beta evaluation metric can be adjustedaccordingly but the cross-entropy objective remains unaware of this preferenceduring training. We propose a method that addresses this training-evaluationgap for multiclass neural network classifiers such that users can train thesemodels informed by the desired final F_beta-Score. Following prior work inbinary classification we utilize the concepts of the soft-set confusionmatrices and a piecewise-linear approximation of the Heaviside step function.Our method extends the 2 times 2 binary soft-set confusion matrix to amulticlass d times d confusion matrix and proposes dynamic adaptation of thethreshold value tau which parameterizes the piecewise-linear Heavisideapproximation during run-time. We present a theoretical analysis that showsthat our method can be used to optimize for a soft-set based approximation ofMacro-F_beta that is a consistent estimator of Macro-F_beta and ourextensive experiments show the practical effectiveness of our approach. |


| Item |Content|
| --- |---|
|idx| 2405.20933v1 |
|title| Concentration Bounds for Optimized Certainty Equivalent Risk Estimation |
|authors| Ayon GhoshL. A. PrashanthKrishna Jagannathan
|links| http://arxiv.org/abs/2405.20933v1 |
|updated| 2024-05-31 15:32:43 UTC |
|summary| We consider the problem of estimating the Optimized Certainty EquivalentOCE risk from independent and identically distributed i.i.d. samples. Forthe classic sample average approximation SAA of OCE we derive mean-squarederror as well as concentration bounds assuming sub-Gaussianity. Further weanalyze an efficient stochastic approximation-based OCE estimator and derivefinite sample bounds for the same. To show the applicability of our bounds weconsider a risk-aware bandit problem with OCE as the risk. For this problemwe derive bound on the probability of mis-identification. Finally we conductnumerical experiments to validate the theoretical findings. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.21044v1 |
|title| Designing for Fairness in Human-Robot Interactions |
|authors| Houston Claure
|links| http://arxiv.org/abs/2405.21044v1 |
|updated| 2024-05-31 17:38:19 UTC |
|summary| The foundation of successful human collaboration is deeply rooted in theprinciples of fairness. As robots are increasingly prevalent in various partsof society where they are working alongside groups and teams of humans theirability to understand and act according to principles of fairness becomescrucial for their effective integration. This is especially critical whenrobots are part of multi-human teams where they must make continuous decisionsregarding the allocation of resources. These resources can be material such astools or communicative such as gaze direction and must be distributed fairlyamong team members to ensure optimal team performance and healthy groupdynamics. Therefore our research focuses on understanding how robots caneffectively participate within human groups by making fair decisions whilecontributing positively to group dynamics and outcomes. In this paper Idiscuss advances toward ensuring that robots are capable of considering humannotions of fairness in their decision-making. |


| Item |Content|
| --- |---|
|idx| 2405.21004v1 |
|title| MunchSonic: Tracking Fine-grained Dietary Actions through Active Acoustic Sensing on Eyeglasses |
|authors| Saif MahmudDevansh AgarwalAshwin AjitQikang LiangThalia VirandaFrancois GuimbretiereCheng Zhang
|links| http://arxiv.org/abs/2405.21004v1 |
|updated| 2024-05-31 16:44:54 UTC |
|summary| We introduce MunchSonic an AI-powered active acoustic sensing systemintegrated into eyeglasses designed to track fine-grained dietary actions likehand-to-mouth movements for food intake chewing and drinking. MunchSonicemits inaudible ultrasonic waves from a commodity eyeglass frame. The reflectedsignals contain rich information about the position and movements of variousbody parts including the mouth jaw arms and hands all of which areinvolved in eating activities. These signals are then processed by a customdeep-learning pipeline to classify six actions: food intake chewing drinkingtalking face-hand touching and other activities null. In an unconstraineduser study with 12 participants MunchSonic achieves a 93.5 macro F1-score ina user-independent evaluation with a 2-second time resolution demonstratingits effectiveness. Additionally MunchSonic accurately tracks eating episodesand the frequency of food intake within those episodes. |


| Item |Content|
| --- |---|
|idx| 2405.20819v1 |
|title| Heuristic evaluations of back support, shoulder support, hand grip strength support, and sit-stand support exoskeletons using universal design principles |
|authors| Alejandra MartinezLaura TovarCarla Irigoyen AmparanKaren GonzalezPrajina EdayathPriyadarshini PennathurArunkumar Pennathur
|links| http://arxiv.org/abs/2405.20819v1 |
|updated| 2024-05-31 14:14:14 UTC |
|summary| Occupational exoskeletons promise to reduce the incidence of musculoskeletalinjuries however we do not know if their designs allow universal use by allworkers. We also do not know how easy the tasks of assembling donningdoffing and disassembling exoskeletons are. The purpose of our study was toheuristically evaluate a back support a shoulder support a handgrip strengthsupport and a sit-stand exoskeleton for how well they are designed foruniversal use when assembling donning doffing and disassembling theexoskeleton. Seven evaluators used universal design principles and associatedcriteria to independently evaluate and rate four exoskeletons when assemblingdonning doffing and disassembling the devices. The rating scale was aLikert-type scale where a rating of 1 represented not at all and a rating of5 represented an excellent design with respect to the universal design criteriafor the task. The results indicate that providing perceptible information tothe user making the design equitable to use for a diverse set of users makingthe design simple and intuitive to use with adequate feedback and designing toprevent user errors and when errors are made allowing the user to recoverquickly from the errors were rated poorly. Assembling and donning taskspresented the most challenges. |


| Item |Content|
| --- |---|
|idx| 2405.20551v1 |
|title| EM-Assist: Safe Automated ExtractMethod Refactoring with LLMs |
|authors| Dorin PomianAbhiram BellurMalinda DilharaZarina KurbatovaEgor BogomolovAndrey SokolovTimofey BryksinDanny Dig
|links| http://dx.doi.org/10.1145/3663529.3663803 |
|updated| 2024-05-31 00:32:04 UTC |
|summary| Excessively long methods loaded with multiple responsibilities arechallenging to understand debug reuse and maintain. The solution lies in thewidely recognized Extract Method refactoring. While the application of thisrefactoring is supported in modern IDEs recommending which code fragments toextract has been the topic of many research tools. However they often struggleto replicate real-world developer practices resulting in recommendations thatdo not align with what a human developer would do in real life. To address thisissue we introduce EM-Assist an IntelliJ IDEA plugin that uses LLMs togenerate refactoring suggestions and subsequently validates enhances andranks them. Finally EM-Assist uses the IntelliJ IDE to apply the user-selectedrecommendation. In our extensive evaluation of 1752 real-world refactoringsthat actually took place in open-source projects EM-Assists recall rate was53.4 among its top-5 recommendations compared to 39.4 for the previousbest-in-class tool that relies solely on static analysis. Moreover weconducted a usability survey with 18 industrial developers and 94.4 gave apositive rating. |


| Item |Content|
| --- |---|
|idx| 2405.20530v1 |
|title| Impact of Connected and Automated Vehicles on Transport Injustices |
|authors| Laura Martinez-BuelvasAndry RakotonirainyDeanna Grant-SmithOscar Oviedo-Trespalacios
|links| http://arxiv.org/abs/2405.20530v1 |
|updated| 2024-05-30 23:07:01 UTC |
|summary| Connected and automated vehicles are poised to transform the transportsystem. However significant uncertainties remain about their impactparticularly regarding concerns that this advanced technology might exacerbateinjustices such as safety disparities for vulnerable road users. Thereforeunderstanding the potential conflicts of this technology with societal valuessuch as justice and safety is crucial for responsible implementation. To dateno research has focused on what safety and justice in transport mean in thecontext of CAV deployment and how the potential benefits of CAVs can beharnessed without exacerbating the existing vulnerabilities and injustices VRUsface. This paper addresses this gap by exploring car drivers and pedestriansperceptions of safety and justice issues that CAVs might exacerbate using anexisting theoretical framework. Employing a qualitative approach the studydelves into the nuanced aspects of these concepts. Interviews were conductedwith 30 participants aged between 18 and 79 in Queensland Australia. Theseinterviews were recorded transcribed organised and analysed using reflexivethematic analysis. Three main themes emerged from the participantsdiscussions: CAVs as a safety problem for VRUs CAVs as a justice problem forVRUs and CAVs as an alignment with societal values problem. Participantsemphasised the safety challenges CAVs pose for VRUs highlighting the need forthorough evaluation and regulatory oversight. Concerns were also raised aboutCAVs potentially marginalising vulnerable groups within society. Participantsadvocated for inclusive discussions and a justice-oriented approach todesigning a comprehensive transport system to address these concerns. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.21027v1 |
|title| Fusion-PSRO: Nash Policy Fusion for Policy Space Response Oracles |
|authors| Jiesong LianYucong HuangMingzhi WangChengdong MaYixue HaoYing WenYaodong Yang
|links| http://arxiv.org/abs/2405.21027v1 |
|updated| 2024-05-31 17:16:29 UTC |
|summary| For solving zero-sum games involving non-transitivity a common approach isto maintain population policies to approximate the Nash Equilibrium NE.Previous research has shown that the Policy Space Response Oracle PSRO is aneffective multi-agent reinforcement learning framework for these games.However repeatedly training new policies from scratch to approximate the BestResponse BR to opponents mixed policies at each iteration is inefficient andcostly. While some PSRO methods initialize a new BR policy by inheriting frompast BR policies this approach limits the exploration of new policiesespecially against challenging opponents.To address this issue we proposeFusion-PSRO which uses model fusion to initialize the policy for betterapproximation to BR. With Top-k probabilities from NE we select high-qualitybase policies and fuse them into a new BR policy through model averaging. Thisapproach allows the initialized policy to incorporate multiple expert policiesmaking it easier to handle difficult opponents compared to inheriting orinitializing from scratch. Additionally our method only modifies the policyinitialization enabling its application to nearly all PSRO variants withoutadditional training overhead.Our experiments with non-transitive matrix gamesLeduc poker and the more complex Liars Dice demonstrate that Fusion-PSROenhances the performance of nearly all PSRO variants achieving lowerexploitability. |


| Item |Content|
| --- |---|
|idx| 2405.20972v1 |
|title| Congestion-Aware Path Re-routing Strategy for Dense Urban Airspace |
|authors| Sajid Ahamed M APrathyush P MenonDebasish Ghose
|links| http://arxiv.org/abs/2405.20972v1 |
|updated| 2024-05-31 16:20:55 UTC |
|summary| Existing UAS Traffic Management UTM frameworks designate preplanned flightpaths to uncrewed aircraft systems UAS enabling the UAS to deliver payloads.However with increasing delivery demand between the source-destination pairsin the urban airspace UAS will likely experience considerable congestion onthe nominal paths. We propose a rule-based congestion mitigation strategy thatimproves UAS safety and airspace utilization in congested traffic streams. Thestrategy relies on nominal path information from the UTM and positionalinformation of other UAS in the vicinity. Following the strategy UAS opts foralternative local paths in the unoccupied airspace surrounding the nominal pathand avoids congested regions. The strategy results in UAS traffic exploring andspreading to alternative adjacent routes on encountering congestion. The paperpresents queuing models to estimate the expected traffic spread for varyingstochastic delivery demand at the source thus helping to reserve the airspacearound the nominal path beforehand to accommodate any foreseen congestion.Simulations are presented to validate the queuing results in the presence ofstatic obstacles and intersecting UAS streams. |


| Item |Content|
| --- |---|
|idx| 2405.20880v1 |
|title| Paying to Do Better: Games with Payments between Learning Agents |
|authors| Yoav KolumbusJoe HalpernÉva Tardos
|links| http://arxiv.org/abs/2405.20880v1 |
|updated| 2024-05-31 14:55:11 UTC |
|summary| In repeated games such as auctions players typically use learningalgorithms to choose their actions. The use of such autonomous learning agentshas become widespread on online platforms. In this paper we explore the impactof players incorporating monetary transfers into their agents algorithmsaiming to incentivize behavior in their favor. Our focus is on understandingwhen players have incentives to make use of monetary transfers how thesepayments affect learning dynamics and what the implications are for welfareand its distribution among the players. We propose a simple game-theoreticmodel to capture such scenarios. Our results on general games show that in abroad class of games players benefit from letting their learning agents makepayments to other learners during the game dynamics and that in many casesthis kind of behavior improves welfare for all players. Our results on first-and second-price auctions show that in equilibria of the payment policygame the agents dynamics can reach strong collusive outcomes with lowrevenue for the auctioneer. These results highlight a challenge for mechanismdesign in systems where automated learning agents can benefit from interactingwith their peers outside the boundaries of the mechanism. |


| Item |Content|
| --- |---|
|idx| 2405.20808v1 |
|title| Optimally Improving Cooperative Learning in a Social Setting |
|authors| Shahrzad HaddadanCheng XinJie Gao
|links| http://arxiv.org/abs/2405.20808v1 |
|updated| 2024-05-31 14:07:33 UTC |
|summary| We consider a cooperative learning scenario where a collection of networkedagents with individually owned classifiers dynamically update theirpredictions for the same classification task through communication orobservations of each others predictions. Clearly if highly influentialvertices use erroneous classifiers there will be a negative effect on theaccuracy of all the agents in the network. We ask the following question: howcan we optimally fix the prediction of a few classifiers so as maximize theoverall accuracy in the entire network. To this end we consider an aggregateand an egalitarian objective function. We show a polynomial time algorithm foroptimizing the aggregate objective function and show that optimizing theegalitarian objective function is NP-hard. Furthermore we developapproximation algorithms for the egalitarian improvement. The performance ofall of our algorithms are guaranteed by mathematical analysis and backed byexperiments on synthetic and real data. |


| Item |Content|
| --- |---|
|idx| 2405.20678v1 |
|title| No-Regret Learning for Fair Multi-Agent Social Welfare Optimization |
|authors| Mengxiao ZhangRamiro Deo-Campo VuongHaipeng Luo
|links| http://arxiv.org/abs/2405.20678v1 |
|updated| 2024-05-31 08:21:11 UTC |
|summary| We consider the problem of online multi-agent Nash social welfare NSWmaximization. While previous works of Hossain et al. 2021 Jones et al.2023 study similar problems in stochastic multi-agent multi-armed bandits andshow that sqrtT-regret is possible after T rounds their fairnessmeasure is the product of all agents rewards instead of their NSW that istheir geometric mean. Given the fundamental role of NSW in the fairnessliterature it is more than natural to ask whether no-regret fair learning withNSW as the objective is possible. In this work we provide a complete answer tothis question in various settings. Specifically in stochastic N-agentK-armed bandits we develop an algorithm withwidetildemathcalOleftKfrac2NTfracN-1Nright regretand prove that the dependence on T is tight making it a sharp contrast tothe sqrtT-regret bounds of Hossain et al. 2021 Jones et al. 2023. Wethen consider a more challenging version of the problem with adversarialrewards. Somewhat surprisingly despite NSW being a concave function we provethat no algorithm can achieve sublinear regret. To circumvent such negativeresults we further consider a setting with full-information feedback anddesign two algorithms with sqrtT-regret: the first one has no dependenceon N at all and is applicable to not just NSW but a broad class of welfarefunctions while the second one has better dependence on K and is preferablewhen N is small. Finally we also show that logarithmic regret is possiblewhenever there exists one agent who is indifferent about different arms. |


