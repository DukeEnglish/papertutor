# cs.CL 

| Item |Content|
| --- |---|
|idx| 2407.07094v1 |
|title| AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning |
|authors| Jiaxi CuiWentao ZhangJing TangXudong TongZhenwei ZhangAmieJing WenRongsheng WangPengfei Wu
|links| http://arxiv.org/abs/2407.07094v1 |
|updated| 2024-07-09 17:59:56 UTC |
|summary| The pervasive deployment of Large Language Models-LLMs in various sectorsoften neglects the nuanced requirements of individuals and small organizationswho benefit more from models precisely tailored to their specific businesscontexts rather than those with broadly superior general capabilities. Thiswork introduces textbfAnyTaskTune a novel fine-tuning methodology coined astextbfTask-Fine-Tune specifically developed to elevate model performance ona diverse array of domain-specific tasks. This method involves a meticulousprocess to identify and define targeted sub-tasks within a domain followed bythe creation of specialized enhancement datasets for fine-tuning therebyoptimizing task-specific model performance. We conducted comprehensivefine-tuning experiments not only in the legal domain for tasks such as keywordextraction and sentence prediction but across over twenty different sub-tasksderived from the domains of finance healthcare law psychology consumerservices and human resources. To substantiate our approach and facilitatecommunity engagement we will open-source these bilingual task datasets. Ourfindings demonstrate that models fine-tuned using the textbfTask-Fine-Tunemethodology not only achieve superior performance on these specific tasks butalso significantly outperform models with higher general capabilities in theirrespective domains. Our work is publicly available aturlhttps://github.com/PandaVT/DataTager. |


| Item |Content|
| --- |---|
|idx| 2407.07093v1 |
|title| FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation |
|authors| Liqun MaMingjie SunZhiqiang Shen
|links| http://arxiv.org/abs/2407.07093v1 |
|updated| 2024-07-09 17:59:48 UTC |
|summary| This work presents a Fully BInarized Large Language Model FBI-LLMdemonstrating for the first time how to train a large-scale binary languagemodel from scratch not the partial binary or ternary LLM like BitNet b1.58 tomatch the performance of its full-precision counterparts e.g. FP16 or BF16in transformer-based LLMs. It achieves this by employing an autoregressivedistillation AD loss with maintaining equivalent model dimensions 130M1.3B 7B and training data volume as regular LLM pretraining while deliveringcompetitive results in terms of perplexity and task-specific effectiveness.Intriguingly by analyzing the training trajectory we find that the pretrainedweight is not necessary for training binarized LLMs from scratch. This researchencourages a new computational framework and may facilitate the future designof specialized hardware tailored for fully 1-bit LLMs. We make all modelscode and training dataset fully accessible and transparent to support furtherresearch Code: https://github.com/LiqunMa/FBI-LLM. Model:https://huggingface.co/LiqunMa/. |


| Item |Content|
| --- |---|
|idx| 2407.07087v1 |
|title| CopyBench: Measuring Literal and Non-Literal Reproduction of Copyright-Protected Text in Language Model Generation |
|authors| Tong ChenAkari AsaiNiloofar MireshghallahSewon MinJames GrimmelmannYejin ChoiHannaneh HajishirziLuke ZettlemoyerPang Wei Koh
|links| http://arxiv.org/abs/2407.07087v1 |
|updated| 2024-07-09 17:58:18 UTC |
|summary| Evaluating the degree of reproduction of copyright-protected content bylanguage models LMs is of significant interest to the AI and legalcommunities. Although both literal and non-literal similarities are consideredby courts when assessing the degree of reproduction prior research has focusedonly on literal similarities. To bridge this gap we introduce CopyBench abenchmark designed to measure both literal and non-literal copying in LMgenerations. Using copyrighted fiction books as text sources we provideautomatic evaluation protocols to assess literal and non-literal copyingbalanced against the model utility in terms of the ability to recall facts fromthe copyrighted works and generate fluent completions. We find that althoughliteral copying is relatively rare two types of non-literal copying -- eventcopying and character copying -- occur even in models as small as 7Bparameters. Larger models demonstrate significantly more copying with literalcopying rates increasing from 0.2 to 10.5 and non-literal copying from 2.3to 6.9 when comparing Llama3-8B and 70B models respectively. We furtherevaluate the effectiveness of current strategies for mitigating copying andshow that 1 training-time alignment can reduce literal copying but mayincrease non-literal copying and 2 current inference-time mitigation methodsprimarily reduce literal but not non-literal copying. |


| Item |Content|
| --- |---|
|idx| 2407.07080v1 |
|title| Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities |
|authors| Shaltiel ShmidmanAvi ShmidmanAmir DN CohenMoshe Koppel
|links| http://arxiv.org/abs/2407.07080v1 |
|updated| 2024-07-09 17:51:37 UTC |
|summary| Training large language models LLMs in low-resource languages such asHebrew poses unique challenges. In this paper we introduce DictaLM2.0 andDictaLM2.0-Instruct two LLMs derived from the Mistral model trained on asubstantial corpus of approximately 200 billion tokens in both Hebrew andEnglish. Adapting a pre-trained model to a new language involves specializedtechniques that differ significantly from training a model from scratch orfurther training existing models on well-resourced languages such as English.We outline these novel training methodologies which facilitate effectivelearning and adaptation to the linguistic properties of Hebrew. Additionallywe fine-tuned DictaLM2.0-Instruct on a comprehensive instruct dataset toenhance its performance on task-specific instructions. To rigorously evaluateour models we introduce a new benchmark suite for Hebrew LLM evaluationcovering a diverse set of tasks including Question Answering SentimentAnalysis Winograd Schema Challenge Translation and Summarization. Our worknot only addresses the intricacies of training LLMs in low-resource languagesbut also proposes a framework that can be leveraged for adapting other LLMs tovarious non-English languages contributing to the broader field ofmultilingual NLP. |


| Item |Content|
| --- |---|
|idx| 2407.07071v1 |
|title| Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps |
|authors| Yung-Sung ChuangLinlu QiuCheng-Yu HsiehRanjay KrishnaYoon KimJames Glass
|links| http://arxiv.org/abs/2407.07071v1 |
|updated| 2024-07-09 17:44:34 UTC |
|summary| When asked to summarize articles or answer questions given a passage largelanguage models LLMs can hallucinate details and respond with unsubstantiatedanswers that are inaccurate with respect to the input context. This paperdescribes a simple approach for detecting such contextual hallucinations. Wehypothesize that contextual hallucinations are related to the extent to whichan LLM attends to information in the provided context versus its owngenerations. Based on this intuition we propose a simple hallucinationdetection model whose input features are given by the ratio of attentionweights on the context versus newly generated tokens for each attention head.We find that a linear classifier based on these lookback ratio features is aseffective as a richer detector that utilizes the entire hidden states of an LLMor a text-based entailment model. The lookback ratio-based detector -- LookbackLens -- is found to transfer across tasks and even models allowing a detectorthat is trained on a 7B model to be applied without retraining to a larger13B model. We further apply this detector to mitigate contextualhallucinations and find that a simple classifier-guided decoding approach isable to reduce the amount of hallucination for example by 9.6 in the XSumsummarization task. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2407.07094v1 |
|title| AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning |
|authors| Jiaxi CuiWentao ZhangJing TangXudong TongZhenwei ZhangAmieJing WenRongsheng WangPengfei Wu
|links| http://arxiv.org/abs/2407.07094v1 |
|updated| 2024-07-09 17:59:56 UTC |
|summary| The pervasive deployment of Large Language Models-LLMs in various sectorsoften neglects the nuanced requirements of individuals and small organizationswho benefit more from models precisely tailored to their specific businesscontexts rather than those with broadly superior general capabilities. Thiswork introduces textbfAnyTaskTune a novel fine-tuning methodology coined astextbfTask-Fine-Tune specifically developed to elevate model performance ona diverse array of domain-specific tasks. This method involves a meticulousprocess to identify and define targeted sub-tasks within a domain followed bythe creation of specialized enhancement datasets for fine-tuning therebyoptimizing task-specific model performance. We conducted comprehensivefine-tuning experiments not only in the legal domain for tasks such as keywordextraction and sentence prediction but across over twenty different sub-tasksderived from the domains of finance healthcare law psychology consumerservices and human resources. To substantiate our approach and facilitatecommunity engagement we will open-source these bilingual task datasets. Ourfindings demonstrate that models fine-tuned using the textbfTask-Fine-Tunemethodology not only achieve superior performance on these specific tasks butalso significantly outperform models with higher general capabilities in theirrespective domains. Our work is publicly available aturlhttps://github.com/PandaVT/DataTager. |


| Item |Content|
| --- |---|
|idx| 2407.07093v1 |
|title| FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation |
|authors| Liqun MaMingjie SunZhiqiang Shen
|links| http://arxiv.org/abs/2407.07093v1 |
|updated| 2024-07-09 17:59:48 UTC |
|summary| This work presents a Fully BInarized Large Language Model FBI-LLMdemonstrating for the first time how to train a large-scale binary languagemodel from scratch not the partial binary or ternary LLM like BitNet b1.58 tomatch the performance of its full-precision counterparts e.g. FP16 or BF16in transformer-based LLMs. It achieves this by employing an autoregressivedistillation AD loss with maintaining equivalent model dimensions 130M1.3B 7B and training data volume as regular LLM pretraining while deliveringcompetitive results in terms of perplexity and task-specific effectiveness.Intriguingly by analyzing the training trajectory we find that the pretrainedweight is not necessary for training binarized LLMs from scratch. This researchencourages a new computational framework and may facilitate the future designof specialized hardware tailored for fully 1-bit LLMs. We make all modelscode and training dataset fully accessible and transparent to support furtherresearch Code: https://github.com/LiqunMa/FBI-LLM. Model:https://huggingface.co/LiqunMa/. |


| Item |Content|
| --- |---|
|idx| 2407.07092v1 |
|title| V-VIPE: Variational View Invariant Pose Embedding |
|authors| Mara LevyAbhinav Shrivastava
|links| http://arxiv.org/abs/2407.07092v1 |
|updated| 2024-07-09 17:59:47 UTC |
|summary| Learning to represent three dimensional 3D human pose given a twodimensional 2D image of a person is a challenging problem. In order to makethe problem less ambiguous it has become common practice to estimate 3D pose inthe camera coordinate space. However this makes the task of comparing two 3Dposes difficult. In this paper we address this challenge by separating theproblem of estimating 3D pose from 2D images into two steps. We use avariational autoencoder VAE to find an embedding that represents 3D poses incanonical coordinate space. We refer to this embedding as variationalview-invariant pose embedding V-VIPE. Using V-VIPE we can encode 2D and 3Dposes and use the embedding for downstream tasks like retrieval andclassification. We can estimate 3D poses from these embeddings using thedecoder as well as generate unseen 3D poses. The variability of our encodingallows it to generalize well to unseen camera views when mapping from 2D space.To the best of our knowledge V-VIPE is the only representation to offer thisdiversity of applications. Code and more information can be found athttps://v-vipe.github.io/. |


| Item |Content|
| --- |---|
|idx| 2407.07088v1 |
|title| Safe and Reliable Training of Learning-Based Aerospace Controllers |
|authors| Udayan MandalGuy AmirHaoze WuIeva DaukantasFletcher Lee NewellUmberto RavaioliBaoluo MengMichael DurlingKerianne HobbsMilan GanaiTobey ShimGuy KatzClark Barrett
|links| http://arxiv.org/abs/2407.07088v1 |
|updated| 2024-07-09 17:58:50 UTC |
|summary| In recent years deep reinforcement learning DRL approaches have generatedhighly successful controllers for a myriad of complex domains. However theopaque nature of these models limits their applicability in aerospace systemsand safety-critical domains in which a single mistake can have direconsequences. In this paper we present novel advancements in both the trainingand verification of DRL controllers which can help ensure their safe behavior.We showcase a design-for-verification approach utilizing k-induction anddemonstrate its use in verifying liveness properties. In addition we also givea brief overview of neural Lyapunov Barrier certificates and summarize theircapabilities on a case study. Finally we describe several other novelreachability-based approaches which despite failing to provide guarantees ofinterest could be effective for verification of other DRL systems and couldbe of further interest to the community. |


| Item |Content|
| --- |---|
|idx| 2407.07086v1 |
|title| Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models |
|authors| Logan CrossViolet XiangAgam BhatiaDaniel LK YaminsNick Haber
|links| http://arxiv.org/abs/2407.07086v1 |
|updated| 2024-07-09 17:57:15 UTC |
|summary| Multi-agent reinforcement learning MARL methods struggle with thenon-stationarity of multi-agent systems and fail to adaptively learn onlinewhen tested with novel agents. Here we leverage large language models LLMsto create an autonomous agent that can handle these challenges. Our agentHypothetical Minds consists of a cognitively-inspired architecture featuringmodular components for perception memory and hierarchical planning over twolevels of abstraction. We introduce the Theory of Mind module that scaffoldsthe high-level planning process by generating hypotheses about other agentsstrategies in natural language. It then evaluates and iteratively refines thesehypotheses by reinforcing hypotheses that make correct predictions about theother agents behavior. Hypothetical Minds significantly improves performanceover previous LLM-agent and RL baselines on a range of competitive mixedmotive and collaborative domains in the Melting Pot benchmark including bothdyadic and population-based environments. Additionally comparisons againstLLM-agent baselines and ablations reveal the importance of hypothesisevaluation and refinement for succeeding on complex scenarios. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2407.07093v1 |
|title| FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation |
|authors| Liqun MaMingjie SunZhiqiang Shen
|links| http://arxiv.org/abs/2407.07093v1 |
|updated| 2024-07-09 17:59:48 UTC |
|summary| This work presents a Fully BInarized Large Language Model FBI-LLMdemonstrating for the first time how to train a large-scale binary languagemodel from scratch not the partial binary or ternary LLM like BitNet b1.58 tomatch the performance of its full-precision counterparts e.g. FP16 or BF16in transformer-based LLMs. It achieves this by employing an autoregressivedistillation AD loss with maintaining equivalent model dimensions 130M1.3B 7B and training data volume as regular LLM pretraining while deliveringcompetitive results in terms of perplexity and task-specific effectiveness.Intriguingly by analyzing the training trajectory we find that the pretrainedweight is not necessary for training binarized LLMs from scratch. This researchencourages a new computational framework and may facilitate the future designof specialized hardware tailored for fully 1-bit LLMs. We make all modelscode and training dataset fully accessible and transparent to support furtherresearch Code: https://github.com/LiqunMa/FBI-LLM. Model:https://huggingface.co/LiqunMa/. |


| Item |Content|
| --- |---|
|idx| 2407.07089v1 |
|title| Fine-Tuning Linear Layers Only Is a Simple yet Effective Way for Task Arithmetic |
|authors| Ruochen JinBojian HouJiancong XiaoWeijie SuLi Shen
|links| http://arxiv.org/abs/2407.07089v1 |
|updated| 2024-07-09 17:59:17 UTC |
|summary| Task arithmetic has recently emerged as a cost-effective and scalableapproach to edit pre-trained models directly in weight space by adding thefine-tuned weights of different tasks. The performance has been furtherimproved by a linear property which is illustrated by weight disentanglement.Yet conventional linearization methods e.g. NTK linearization not onlydouble the time and training cost but also have a disadvantage on single-taskperformance. We propose a simple yet effective and efficient method that onlyfine-tunes linear layers which improves weight disentanglement and efficiencysimultaneously. Specifically our study reveals that only fine-tuning thelinear layers in the attention modules makes the whole model occur in a linearregime significantly improving weight disentanglement. To further understandhow our method improves the disentanglement of task arithmetic we present acomprehensive study of task arithmetic by differentiating the role ofrepresentation model and task-specific model. In particular we find that therepresentation model plays an important role in improving weightdisentanglement whereas the task-specific models such as the classificationheads can degenerate the weight disentanglement performance. Overall our workuncovers novel insights into the fundamental mechanisms of task arithmetic andoffers a more reliable and effective approach to editing pre-trained models. |


| Item |Content|
| --- |---|
|idx| 2407.07087v1 |
|title| CopyBench: Measuring Literal and Non-Literal Reproduction of Copyright-Protected Text in Language Model Generation |
|authors| Tong ChenAkari AsaiNiloofar MireshghallahSewon MinJames GrimmelmannYejin ChoiHannaneh HajishirziLuke ZettlemoyerPang Wei Koh
|links| http://arxiv.org/abs/2407.07087v1 |
|updated| 2024-07-09 17:58:18 UTC |
|summary| Evaluating the degree of reproduction of copyright-protected content bylanguage models LMs is of significant interest to the AI and legalcommunities. Although both literal and non-literal similarities are consideredby courts when assessing the degree of reproduction prior research has focusedonly on literal similarities. To bridge this gap we introduce CopyBench abenchmark designed to measure both literal and non-literal copying in LMgenerations. Using copyrighted fiction books as text sources we provideautomatic evaluation protocols to assess literal and non-literal copyingbalanced against the model utility in terms of the ability to recall facts fromthe copyrighted works and generate fluent completions. We find that althoughliteral copying is relatively rare two types of non-literal copying -- eventcopying and character copying -- occur even in models as small as 7Bparameters. Larger models demonstrate significantly more copying with literalcopying rates increasing from 0.2 to 10.5 and non-literal copying from 2.3to 6.9 when comparing Llama3-8B and 70B models respectively. We furtherevaluate the effectiveness of current strategies for mitigating copying andshow that 1 training-time alignment can reduce literal copying but mayincrease non-literal copying and 2 current inference-time mitigation methodsprimarily reduce literal but not non-literal copying. |


| Item |Content|
| --- |---|
|idx| 2407.07084v1 |
|title| Stabilized Proximal-Point Methods for Federated Optimization |
|authors| Xiaowen JiangAnton RodomanovSebastian U. Stich
|links| http://arxiv.org/abs/2407.07084v1 |
|updated| 2024-07-09 17:56:29 UTC |
|summary| In developing efficient optimization algorithms it is crucial to account forcommunication constraints -- a significant challenge in modern federatedlearning settings. The best-known communication complexity amongnon-accelerated algorithms is achieved by DANE a distributed proximal-pointalgorithm that solves local subproblems in each iteration and that can exploitsecond-order similarity among individual functions. However to achieve suchcommunication efficiency the accuracy requirement for solving the localsubproblems is slightly sub-optimal. Inspired by the hybrid projection-proximalpoint method in this work we i propose a novel distributed algorithm S-DANE.This method adopts a more stabilized prox-center in the proximal step comparedwith DANE and matches its deterministic communication complexity. Moreoverthe accuracy condition of the subproblem is milder leading to enhanced localcomputation efficiency. Furthermore it supports partial client participationand arbitrary stochastic local solvers making it more attractive in practice.We further ii accelerate S-DANE and show that the resulting algorithmachieves the best-known communication complexity among all existing methods fordistributed convex optimization with the same improved local computationefficiency as S-DANE. |


| Item |Content|
| --- |---|
|idx| 2407.07082v1 |
|title| Can Learned Optimization Make Reinforcement Learning Less Difficult? |
|authors| Alexander David GoldieChris LuMatthew Thomas JacksonShimon WhitesonJakob Nicolaus Foerster
|links| http://arxiv.org/abs/2407.07082v1 |
|updated| 2024-07-09 17:55:23 UTC |
|summary| While reinforcement learning RL holds great potential for decision makingin the real world it suffers from a number of unique difficulties which oftenneed specific consideration. In particular: it is highly non-stationarysuffers from high degrees of plasticity loss and requires exploration toprevent premature convergence to local optima and maximize return. In thispaper we consider whether learned optimization can help overcome theseproblems. Our method Learned Optimization for Plasticity Exploration andNon-stationarity OPEN meta-learns an update rule whose input features andoutput structure are informed by previously proposed solutions to thesedifficulties. We show that our parameterization is flexible enough to enablemeta-learning in diverse learning contexts including the ability to usestochasticity for exploration. Our experiments demonstrate that whenmeta-trained on single and small sets of environments OPEN outperforms orequals traditionally used optimizers. Furthermore OPEN shows stronggeneralization across a distribution of environments and a range of agentarchitectures. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2407.07092v1 |
|title| V-VIPE: Variational View Invariant Pose Embedding |
|authors| Mara LevyAbhinav Shrivastava
|links| http://arxiv.org/abs/2407.07092v1 |
|updated| 2024-07-09 17:59:47 UTC |
|summary| Learning to represent three dimensional 3D human pose given a twodimensional 2D image of a person is a challenging problem. In order to makethe problem less ambiguous it has become common practice to estimate 3D pose inthe camera coordinate space. However this makes the task of comparing two 3Dposes difficult. In this paper we address this challenge by separating theproblem of estimating 3D pose from 2D images into two steps. We use avariational autoencoder VAE to find an embedding that represents 3D poses incanonical coordinate space. We refer to this embedding as variationalview-invariant pose embedding V-VIPE. Using V-VIPE we can encode 2D and 3Dposes and use the embedding for downstream tasks like retrieval andclassification. We can estimate 3D poses from these embeddings using thedecoder as well as generate unseen 3D poses. The variability of our encodingallows it to generalize well to unseen camera views when mapping from 2D space.To the best of our knowledge V-VIPE is the only representation to offer thisdiversity of applications. Code and more information can be found athttps://v-vipe.github.io/. |


| Item |Content|
| --- |---|
|idx| 2407.07090v1 |
|title| 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes |
|authors| Nicolas Moenne-LoccozAshkan MirzaeiOr PerelRiccardo de LutioJanick Martinez EsturoGavriel StateSanja FidlerNicholas SharpZan Gojcic
|links| http://arxiv.org/abs/2407.07090v1 |
|updated| 2024-07-09 17:59:30 UTC |
|summary| Particle-based representations of radiance fields such as 3D GaussianSplatting have found great success for reconstructing and re-rendering ofcomplex scenes. Most existing methods render particles via rasterizationprojecting them to screen space tiles for processing in a sorted order. Thiswork instead considers ray tracing the particles building a bounding volumehierarchy and casting a ray for each pixel using high-performance GPU raytracing hardware. To efficiently handle large numbers of semi-transparentparticles we describe a specialized rendering algorithm which encapsulatesparticles with bounding meshes to leverage fast ray-triangle intersections andshades batches of intersections in depth-order. The benefits of ray tracing arewell-known in computer graphics: processing incoherent rays for secondarylighting effects such as shadows and reflections rendering fromhighly-distorted cameras common in robotics stochastically sampling rays andmore. With our renderer this flexibility comes at little cost compared torasterization. Experiments demonstrate the speed and accuracy of our approachas well as several applications in computer graphics and vision. We furtherpropose related improvements to the basic Gaussian representation including asimple use of generalized kernel functions which significantly reduces particlehit counts. |


| Item |Content|
| --- |---|
|idx| 2407.07078v1 |
|title| MoSt-DSA: Modeling Motion and Structural Interactions for Direct Multi-Frame Interpolation in DSA Images |
|authors| Ziyang XuHuangxuan ZhaoZiwei CuiWenyu LiuChuansheng ZhengXinggang Wang
|links| http://arxiv.org/abs/2407.07078v1 |
|updated| 2024-07-09 17:50:54 UTC |
|summary| Artificial intelligence has become a crucial tool for medical image analysis.As an advanced cerebral angiography technique Digital Subtraction AngiographyDSA poses a challenge where the radiation dose to humans is proportional tothe image count. By reducing images and using AI interpolation instead theradiation can be cut significantly. However DSA images present more complexmotion and structural features than natural scenes making interpolation morechallenging. We propose MoSt-DSA the first work that uses deep learning forDSA frame interpolation. Unlike natural scene Video Frame Interpolation VFImethods that extract unclear or coarse-grained features we devise a generalmodule that models motion and structural context interactions between frames inan efficient full convolution manner by adjusting optimal context range andtransforming contexts into linear functions. Benefiting from this MoSt-DSA isalso the first method that directly achieves any number of interpolations atany time steps with just one forward pass during both training and testing. Weconduct extensive comparisons with 7 representative VFI models forinterpolating 1 to 3 frames MoSt-DSA demonstrates robust results across 470DSA image sequences each typically 152 images with average SSIM over 0.93average PSNR over 38 standard deviations of less than 0.030 and 3.6respectively comprehensively achieving state-of-the-art performance inaccuracy speed visual effect and memory usage. Our code is available athttps://github.com/ZyoungXu/MoSt-DSA. |


| Item |Content|
| --- |---|
|idx| 2407.07077v1 |
|title| ConceptExpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction |
|authors| Shaozhe HaoKai HanZhengyao LvShihao ZhaoKwan-Yee K. Wong
|links| http://arxiv.org/abs/2407.07077v1 |
|updated| 2024-07-09 17:50:28 UTC |
|summary| While personalized text-to-image generation has enabled the learning of asingle concept from multiple images a more practical yet challenging scenarioinvolves learning multiple concepts within a single image. However existingworks tackling this scenario heavily rely on extensive human annotations. Inthis paper we introduce a novel task named Unsupervised Concept ExtractionUCE that considers an unsupervised setting without any human knowledge of theconcepts. Given an image that contains multiple concepts the task aims toextract and recreate individual concepts solely relying on the existingknowledge from pretrained diffusion models. To achieve this we presentConceptExpress that tackles UCE by unleashing the inherent capabilities ofpretrained diffusion models in two aspects. Specifically a conceptlocalization approach automatically locates and disentangles salient conceptsby leveraging spatial correspondence from diffusion self-attention and basedon the lookup association between a concept and a conceptual token aconcept-wise optimization process learns discriminative tokens that representeach individual concept. Finally we establish an evaluation protocol tailoredfor the UCE task. Extensive experiments demonstrate that ConceptExpress is apromising solution to the UCE task. Our code and data are available at:https://github.com/haoosz/ConceptExpress |


| Item |Content|
| --- |---|
|idx| 2407.07076v1 |
|title| MADE-for-ASD: A Multi-Atlas Deep Ensemble Network for Diagnosing Autism Spectrum Disorder |
|authors| Md Rakibul HasanXuehan LiuTom GedeonMd Zakir Hossain
|links| http://arxiv.org/abs/2407.07076v1 |
|updated| 2024-07-09 17:49:23 UTC |
|summary| In response to the global need for efficient early diagnosis of AutismSpectrum Disorder ASD this paper bridges the gap between traditionaltime-consuming diagnostic methods and potential automated solutions. We proposea multi-atlas deep ensemble network MADE-for-ASD that integrates multipleatlases of the brains functional magnetic resonance imaging fMRI datathrough a weighted deep ensemble network. Our approach integrates demographicinformation into the prediction workflow which enhances ASD diagnosisperformance and offers a more holistic perspective on patient profiling. Weexperiment with the well-known publicly available ABIDE Autism Brain ImagingData Exchange I dataset consisting of resting state fMRI data from 17different laboratories around the globe. Our proposed system achieves 75.20accuracy on the entire dataset and 96.40 on a specific subset - bothsurpassing reported ASD diagnosis accuracy in ABIDE I fMRI studies.Specifically our model improves by 4.4 percentage points over prior works onthe same amount of data. The model exhibits a sensitivity of 82.90 and aspecificity of 69.70 on the entire dataset and 91.00 and 99.50respectively on the specific subset. We leverage the F-score to pinpoint thetop 10 ROI in ASD diagnosis such as emphprecuneus and anterioremphcingulate/ventromedial. The proposed system can potentially pave the wayfor more cost-effective efficient and scalable strategies in ASD diagnosis.Codes and evaluations are publicly available at TBA. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2407.06945v1 |
|title| Adaptively Robust and Sparse K-means Clustering |
|authors| Hao LiShonosuke SugasawaShota Katayama
|links| http://arxiv.org/abs/2407.06945v1 |
|updated| 2024-07-09 15:20:41 UTC |
|summary| While K-means is known to be a standard clustering algorithm it may becompromised due to the presence of outliers and high-dimensional noisyvariables. This paper proposes adaptively robust and sparse K-means clusteringARSK to address these practical limitations of the standard K-meansalgorithm. We introduce a redundant error component for each observation forrobustness and this additional parameter is penalized using a group sparsepenalty. To accommodate the impact of high-dimensional noisy variables theobjective function is modified by incorporating weights and implementing apenalty to control the sparsity of the weight vector. The tuning parameters tocontrol the robustness and sparsity are selected by Gap statistics. Throughsimulation experiments and real data analysis we demonstrate the superiorityof the proposed method to existing algorithms in identifying clusters withoutoutliers and informative variables simultaneously. |


| Item |Content|
| --- |---|
|idx| 2407.06935v1 |
|title| Bayesian Federated Learning with Hamiltonian Monte Carlo: Algorithm and Theory |
|authors| Jiajun LiangQian ZhangWei DengQifan SongGuang Lin
|links| http://arxiv.org/abs/2407.06935v1 |
|updated| 2024-07-09 15:10:59 UTC |
|summary| This work introduces a novel and efficient Bayesian federated learningalgorithm namely the Federated Averaging stochastic Hamiltonian Monte CarloFA-HMC for parameter estimation and uncertainty quantification. We establishrigorous convergence guarantees of FA-HMC on non-iid distributed data setsunder the strong convexity and Hessian smoothness assumptions. Our analysisinvestigates the effects of parameter space dimension noise on gradients andmomentum and the frequency of communication between the central node andlocal nodes on the convergence and communication costs of FA-HMC. Beyond thatwe establish the tightness of our analysis by showing that the convergence ratecannot be improved even for continuous FA-HMC process. Moreover extensiveempirical studies demonstrate that FA-HMC outperforms the existing FederatedAveraging-Langevin Monte Carlo FA-LD algorithm. |


| Item |Content|
| --- |---|
|idx| 2407.06867v1 |
|title| Distributionally robust risk evaluation with an isotonic constraint |
|authors| Yu GuiRina Foygel BarberCong Ma
|links| http://arxiv.org/abs/2407.06867v1 |
|updated| 2024-07-09 13:56:34 UTC |
|summary| Statistical learning under distribution shift is challenging when neitherprior knowledge nor fully accessible data from the target distribution isavailable. Distributionally robust learning DRL aims to control theworst-case statistical performance within an uncertainty set of candidatedistributions but how to properly specify the set remains challenging. Toenable distributional robustness without being overly conservative in thispaper we propose a shape-constrained approach to DRL which incorporates priorinformation about the way in which the unknown target distribution differs fromits estimate. More specifically we assume the unknown density ratio betweenthe target distribution and its estimate is isotonic with respect to somepartial order. At the population level we provide a solution to theshape-constrained optimization problem that does not involve the isotonicconstraint. At the sample level we provide consistency results for anempirical estimator of the target in a range of different settings. Empiricalstudies on both synthetic and real data examples demonstrate the improvedaccuracy of the proposed shape-constrained approach. |


| Item |Content|
| --- |---|
|idx| 2407.06797v1 |
|title| ED-VAE: Entropy Decomposition of ELBO in Variational Autoencoders |
|authors| Fotios LygerakisElmar Rueckert
|links| http://arxiv.org/abs/2407.06797v1 |
|updated| 2024-07-09 12:09:21 UTC |
|summary| Traditional Variational Autoencoders VAEs are constrained by thelimitations of the Evidence Lower Bound ELBO formulation particularly whenutilizing simplistic non-analytic or unknown prior distributions. Theselimitations inhibit the VAEs ability to generate high-quality samples andprovide clear interpretable latent representations. This work introduces theEntropy Decomposed Variational Autoencoder ED-VAE a novel re-formulation ofthe ELBO that explicitly includes entropy and cross-entropy components. Thisreformulation significantly enhances model flexibility allowing for theintegration of complex and non-standard priors. By providing more detailedcontrol over the encoding and regularization of latent spaces ED-VAE not onlyimproves interpretability but also effectively captures the complexinteractions between latent variables and observed data thus leading to bettergenerative performance. |


| Item |Content|
| --- |---|
|idx| 2407.06765v1 |
|title| A Generalization Bound for Nearly-Linear Networks |
|authors| Eugene Golikov
|links| http://arxiv.org/abs/2407.06765v1 |
|updated| 2024-07-09 11:20:01 UTC |
|summary| We consider nonlinear networks as perturbations of linear ones. Based on thisapproach we present novel generalization bounds that become non-vacuous fornetworks that are close to being linear. The main advantage over the previousworks which propose non-vacuous generalization bounds is that our bounds area-priori: performing the actual training is not required for evaluating thebounds. To the best of our knowledge they are the first non-vacuousgeneralization bounds for neural nets possessing this property. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2407.07040v1 |
|title| Garment suggestion based on comfort extracted from physiological and emotional parameters |
|authors| Hyo JungChangMohammad Abu Nasir RakibKamrul H FoysalJo Woon Chong
|links| http://arxiv.org/abs/2407.07040v1 |
|updated| 2024-07-09 17:02:33 UTC |
|summary| The purpose of the study was to find the true comfort of the wearer byconceptualizing formulating and proving the relation between physiologicaland emotional parameters with clothing fit and fabric. A mixed-method researchdesign was used and the findings showed that physiological indicators such asheart rate are closely linked with user comfort. However a significant changein emotional response indicated a definite relationship between differentfabric and fit types. The research was conducted to discover the relationbetween true comfort parameters and clothing which is unique to the field. Thefindings help us understand how fabric types and clothing fit types can affectphysiological and emotional responses providing the consumer with satisfactoryclothing with the suitable properties needed. |


| Item |Content|
| --- |---|
|idx| 2407.07015v1 |
|title| A Framework for Multimodal Medical Image Interaction |
|authors| Laura SchützSasan MatinfarGideon SchafrothNavid NavabMerle FairhurstArthur WagnerBenedikt WiestlerUlrich EckNassir Navab
|links| http://arxiv.org/abs/2407.07015v1 |
|updated| 2024-07-09 16:33:51 UTC |
|summary| Medical doctors rely on images of the human anatomy such as magneticresonance imaging MRI to localize regions of interest in the patient duringdiagnosis and treatment. Despite advances in medical imaging technology theinformation conveyance remains unimodal. This visual representation fails tocapture the complexity of the real multisensory interaction with human tissue.However perceiving multimodal information about the patients anatomy anddisease in real-time is critical for the success of medical procedures andpatient outcome. We introduce a Multimodal Medical Image Interaction MMIIframework to allow medical experts a dynamic audiovisual interaction withhuman tissue in three-dimensional space. In a virtual reality environment theuser receives physically informed audiovisual feedback to improve the spatialperception of anatomical structures. MMII uses a model-based sonificationapproach to generate sounds derived from the geometry and physical propertiesof tissue thereby eliminating the need for hand-crafted sound design. Two userstudies involving 34 general and nine clinical experts were conducted toevaluate the proposed interaction frameworks learnability usability andaccuracy. Our results showed excellent learnability of audiovisualcorrespondence as the rate of correct associations significantly improved p 0.001 over the course of the study. MMII resulted in superior brain tumorlocalization accuracy p  0.05 compared to conventional medical imageinteraction. Our findings substantiate the potential of this novel framework toenhance interaction with medical images for example during surgicalprocedures where immediate and precise feedback is needed. |


| Item |Content|
| --- |---|
|idx| 2407.06972v1 |
|title| Microsoft Cloud-based Digitization Workflow with Rich Metadata Acquisition for Cultural Heritage Objects |
|authors| Krzysztof KuttJakub GomułkaLuiz do Valle MirandaGrzegorz J. Nalepa
|links| http://arxiv.org/abs/2407.06972v1 |
|updated| 2024-07-09 15:49:47 UTC |
|summary| In response to several cultural heritage initiatives at the JagiellonianUniversity we have developed a new digitization workflow in collaboration withthe Jagiellonian Library JL. The solution is based on easy-to-accesstechnological solutions -- Microsoft 365 cloud with MS Excel files as metadataacquisition interfaces Office Script for validation and MS Sharepoint forstorage -- that allows metadata acquisition by domain experts philologistshistorians philosophers librarians archivists curators etc. regardless oftheir experience with information systems. The ultimate goal is to create aknowledge graph that describes the analyzed holdings linked to generalknowledge bases as well as to other cultural heritage collections so carefulattention is paid to the high accuracy of metadata and proper links to externalsources. The workflow has already been evaluated in two pilots in the DiHeLibproject focused on digitizing the so-called Berlin Collection and in twoworkshops with international guests which allowed for its refinement andconfirmation of its correctness and usability for JL. As the proposed workflowdoes not interfere with existing systems or domain guidelines regardingdigitization and basic metadata collection in a given institution e.g. filetype image quality use of Dublin Core/MARC-21 but extends them in order toenable rich metadata collection not previously possible we believe that itcould be of interest to all GLAMs galleries libraries archives andmuseums. |


| Item |Content|
| --- |---|
|idx| 2407.06967v1 |
|title| INTERACT: An authoring tool that facilitates the creation of human centric interaction with 3d objects in virtual reality |
|authors| Rama Krishnan Gopal Ramasamy ThandapaniBenjamin CapelAntoine LasnierIoannis Chatzigiannakis
|links| http://arxiv.org/abs/2407.06967v1 |
|updated| 2024-07-09 15:46:52 UTC |
|summary| A widespread adoption of Virtual Augmented and Mixed Reality VR/AR/MRcollectively referred to as Extended Reality XR has become a tangiblepossibility to revolutionize educational and training scenarios by offeringimmersive interactive experiences. In this paper we present textsfINTERACTan authoring tool for creating advanced 3D physics-based Intelligent TutoringSystems ITS by individual developers or small-scale development teams.textsfINTERACT is based on a cutting edge physics engine allowing realisticinteractions such as collision detection and ergonomic evaluations. Wedemonstrate the benefits of textsfINTERACT by developing a set of trainingscenarios for a use case of a Laser cutting machine. The use case illustratesthe numerous possibilities such as creating interaction with objects ease ofconfiguring a scenario and how to design the visual effects to the machine. |


| Item |Content|
| --- |---|
|idx| 2407.06902v1 |
|title| Learning From Crowdsourced Noisy Labels: A Signal Processing Perspective |
|authors| Shahana IbrahimPanagiotis A. TraganitisXiao FuGeorgios B. Giannakis
|links| http://arxiv.org/abs/2407.06902v1 |
|updated| 2024-07-09 14:34:40 UTC |
|summary| One of the primary catalysts fueling advances in artificial intelligence AIand machine learning ML is the availability of massive curated datasets. Acommonly used technique to curate such massive datasets is crowdsourcing wheredata are dispatched to multiple annotators. The annotator-produced labels arethen fused to serve downstream learning and inference tasks. This annotationprocess often creates noisy labels due to various reasons such as the limitedexpertise or unreliability of annotators among others. Therefore a coreobjective in crowdsourcing is to develop methods that effectively mitigate thenegative impact of such label noise on learning tasks. This feature articleintroduces advances in learning from noisy crowdsourced labels. The focus is onkey crowdsourcing models and their methodological treatments from classicalstatistical models to recent deep learning-based approaches emphasizinganalytical insights and algorithmic developments. In particular this articlereviews the connections between signal processing SP theory and methods suchas identifiability of tensor and nonnegative matrix factorization and novelprincipled solutions of longstanding challenges in crowdsourcing -- showing howSP perspectives drive the advancements of this field. Furthermore this articletouches upon emerging topics that are critical for developing cutting-edgeAI/ML systems such as crowdsourcing in reinforcement learning with humanfeedback RLHF and direct preference optimization DPO that are keytechniques for fine-tuning large language models LLMs. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2407.06886v1 |
|title| Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI |
|authors| Yang LiuWeixing ChenYongjie BaiJingzhou LuoXinshuai SongKaixuan JiangZhida LiGanlong ZhaoJunyi LinGuanbin LiWen GaoLiang Lin
|links| http://arxiv.org/abs/2407.06886v1 |
|updated| 2024-07-09 14:14:47 UTC |
|summary| Embodied Artificial Intelligence Embodied AI is crucial for achievingArtificial General Intelligence AGI and serves as a foundation for variousapplications that bridge cyberspace and the physical world. Recently theemergence of Multi-modal Large Models MLMs and World Models WMs haveattracted significant attention due to their remarkable perceptioninteraction and reasoning capabilities making them a promising architecturefor the brain of embodied agents. However there is no comprehensive survey forEmbodied AI in the era of MLMs. In this survey we give a comprehensiveexploration of the latest advancements in Embodied AI. Our analysis firstlynavigates through the forefront of representative works of embodied robots andsimulators to fully understand the research focuses and their limitations.Then we analyze four main research targets: 1 embodied perception 2embodied interaction 3 embodied agent and 4 sim-to-real adaptationcovering the state-of-the-art methods essential paradigms and comprehensivedatasets. Additionally we explore the complexities of MLMs in virtual and realembodied agents highlighting their significance in facilitating interactionsin dynamic digital and physical environments. Finally we summarize thechallenges and limitations of embodied AI and discuss their potential futuredirections. We hope this survey will serve as a foundational reference for theresearch community and inspire continued innovation. The associated project canbe found at https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List. |


| Item |Content|
| --- |---|
|idx| 2407.06813v1 |
|title| Richelieu: Self-Evolving LLM-Based Agents for AI Diplomacy |
|authors| Zhenyu GuanXiangyu KongFangwei ZhongYizhou Wang
|links| http://arxiv.org/abs/2407.06813v1 |
|updated| 2024-07-09 12:37:54 UTC |
|summary| Diplomacy is one of the most sophisticated activities in human society. Thecomplex interactions among multiple parties/ agents involve various abilitieslike social reasoning negotiation arts and long-term strategy planning.Previous AI agents surely have proved their capability of handling multi-stepgames and larger action spaces on tasks involving multiple agents. Howeverdiplomacy involves a staggering magnitude of decision spaces especiallyconsidering the negotiation stage required. Recently LLM agents have showntheir potential for extending the boundary of previous agents on a couple ofapplications however it is still not enough to handle a very long planningperiod in a complex multi-agent environment. Empowered with cutting-edge LLMtechnology we make the first stab to explore AIs upper bound towards ahuman-like agent for such a highly comprehensive multi-agent mission bycombining three core and essential capabilities for stronger LLM-based societalagents: 1 strategic planner with memory and reflection 2 goal-orientednegotiate with social reasoning 3 augmenting memory by self-play games toself-evolving without any human in the loop. |


| Item |Content|
| --- |---|
|idx| 2407.06541v1 |
|title| Fast Distributed Optimization over Directed Graphs under Malicious Attacks using Trust |
|authors| Arif Kerem DayıOrhan Eren AkgünStephanie GilMichal YeminiAngelia Nedić
|links| http://arxiv.org/abs/2407.06541v1 |
|updated| 2024-07-09 04:22:35 UTC |
|summary| In this work we introduce the Resilient Projected Push-Pull RP3 algorithmdesigned for distributed optimization in multi-agent cyber-physical systemswith directed communication graphs and the presence of malicious agents. Ouralgorithm leverages stochastic inter-agent trust values and gradient trackingto achieve geometric convergence rates in expectation even in adversarialenvironments. We introduce growing constraint sets to limit the impact of themalicious agents without compromising the geometric convergence rate of thealgorithm. We prove that RP3 converges to the nominal optimal solution almostsurely and in the r-th mean for any rgeq 1 provided the step sizes aresufficiently small and the constraint sets are appropriately chosen. Wevalidate our approach with numerical studies on average consensus andmulti-robot target tracking problems demonstrating that RP3 effectivelymitigates the impact of malicious agents and achieves the desired geometricconvergence. |


| Item |Content|
| --- |---|
|idx| 2407.06454v1 |
|title| Analysis of Robotic System Models Through Property Inheritance from Petri Net Meta-models |
|authors| Maksym FigatCezary Zieliński
|links| http://arxiv.org/abs/2407.06454v1 |
|updated| 2024-07-08 23:35:36 UTC |
|summary| This article investigates the analysis of robotic system models using theRobotic System Hierarchic Petri Net RSHPN meta-model proposing streamlinedmethods by focusing on significant system fragments and inheriting propertiesfrom the meta-model. Our research demonstrates that it is feasible to: 1effectively analyze complex robotic systems expressed using RSHPN and 2enable models to inherit properties from the meta-model. This approachsignificantly simplifies the analysis process reduces design time and ensuresthe safety and reliability of the systems. These aspects are crucial for robotsoperating in human environments. Our results suggest that Petri nets could befurther explored as a useful tool for the formal description and in-depthanalysis of the properties of robotic systems. |


| Item |Content|
| --- |---|
|idx| 2407.06426v1 |
|title| DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations |
|authors| Luke YoffeAlfonso AmayuelasWilliam Yang Wang
|links| http://arxiv.org/abs/2407.06426v1 |
|updated| 2024-07-08 22:15:01 UTC |
|summary| To enhance Large Language Model LLM capabilities multi-agent debates havebeen introduced where multiple LLMs discuss solutions to a problem overseveral rounds of debate. However LLMs often produce incorrect responses thatappear deceptively confident which can mislead other agents. This is partlybecause agents do not express their confidence levels during standard debates.To address this we introduce DebUnc a multi-agent debate framework that usesuncertainty metrics to assess agent confidence levels. We adapted the LLMattention mechanism to adjust token weights based on confidence levels and alsoexplored using textual prompts to convey confidence. Our evaluations acrossvarious benchmarks show that attention-based methods are particularlyeffective and that as uncertainty metrics evolve performance will continue toincrease. The code is available at https://github.com/lukeyoffe/debunc |


