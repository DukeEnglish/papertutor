# cs.CL 

| Item |Content|
| --- |---|
|idx| 2411.09694v1 |
|title| A Bayesian Optimization Approach to Machine Translation Reranking |
|authors| Julius ChengMaike ZüfleVilém ZouharAndreas Vlachos
|links| http://arxiv.org/abs/2411.09694v1 |
|updated| 2024-11-14 18:58:23 UTC |
|summary| Reranking a list of candidates from a machine translation system with anexternal scoring model and returning the highest-scoring candidate remains asimple and effective method for improving the overall output quality.Translation scoring models continue to grow in size with the best models beingcomparable to generation models. Thus reranking can add substantialcomputational cost to the translation pipeline. In this work we pose rerankingas a Bayesian optimization BayesOpt problem. By strategically selectingcandidates to score based on a balance of exploration and exploitation we showthat it is possible to find top-scoring candidates when scoring only a fractionof the candidate list. For instance our method achieves the same CometKiwiscore using only 70 scoring evaluations compared a baseline system using 180.We present a multi-fidelity setting for BayesOpt where the candidates arefirst scored with a cheaper but noisier proxy scoring model which furtherimproves the cost-performance tradeoff when using smaller but well-traineddistilled proxy scorers. |


| Item |Content|
| --- |---|
|idx| 2411.09689v1 |
|title| LLM Hallucination Reasoning with Zero-shot Knowledge Test |
|authors| Seongmin LeeHsiang HsuChun-Fu Chen
|links| http://arxiv.org/abs/2411.09689v1 |
|updated| 2024-11-14 18:55:26 UTC |
|summary| LLM hallucination where LLMs occasionally generate unfaithful text posessignificant challenges for their practical applications. Most existingdetection methods rely on external knowledge LLM fine-tuning orhallucination-labeled datasets and they do not distinguish between differenttypes of hallucinations which are crucial for improving detection performance.We introduce a new task Hallucination Reasoning which classifiesLLM-generated text into one of three categories: aligned misaligned andfabricated. Our novel zero-shot method assesses whether LLM has enoughknowledge about a given prompt and text. Our experiments conducted on newdatasets demonstrate the effectiveness of our method in hallucination reasoningand underscore its importance for enhancing detection performance. |


| Item |Content|
| --- |---|
|idx| 2411.09688v1 |
|title| Squeezed Attention: Accelerating Long Context Length LLM Inference |
|authors| Coleman HooperSehoon KimHiva MohammadzadehMonishwaran MaheswaranJune PaikMichael W. MahoneyKurt KeutzerAmir Gholami
|links| http://arxiv.org/abs/2411.09688v1 |
|updated| 2024-11-14 18:54:19 UTC |
|summary| Emerging Large Language Model LLM applications require long input promptsto perform complex downstream tasks like document analysis and code generation.For these long context length applications the length of the input promptposes a significant challenge in terms of inference efficiency since theinference costs increase linearly with sequence length. However for many ofthese applications much of the context in the prompt is fixed across differentuser inputs thereby providing the opportunity to perform offline optimizationsto process user inputs quickly as they are received. In this work we proposeSqueezed Attention as a mechanism to accelerate LLM applications where a largeportion of the input prompt is fixed. We first leverage K-means clusteringoffline to group the keys for the fixed context based on semantic similarityand represent each cluster with a single centroid value. During inference wecompare query tokens from the user input with the centroids to predict which ofthe keys from the fixed context are semantically relevant and need to be loadedduring inference. We then compute exact attention using only these importantkeys from the fixed context thereby reducing bandwidth and computationalcosts. We also extend our method to use a hierarchical centroid lookup toidentify important keys which can reduce the complexity of attention fromlinear to logarithmic with respect to the context length. We implementoptimized Triton kernels for centroid comparison and sparse FlashAttention withimportant keys achieving more than 4x speedups during both the prefill andgeneration phases for long-context inference. Furthermore we have extensivelyevaluated our method on various long-context benchmarks including LongBenchwhere it achieves a 3x reduction in KV cache budget without accuracy loss andup to an 8x reduction with 0.5 point accuracy gap for various models. |


| Item |Content|
| --- |---|
|idx| 2411.09661v1 |
|title| Adaptive Decoding via Latent Preference Optimization |
|authors| Shehzaad DhuliawalaIlia KulikovPing YuAsli CelikyilmazJason WestonSainbayar SukhbaatarJack Lanchantin
|links| http://arxiv.org/abs/2411.09661v1 |
|updated| 2024-11-14 18:31:39 UTC |
|summary| During language model decoding it is known that using higher temperaturesampling gives more creative responses while lower temperatures are morefactually accurate. However such models are commonly applied to generalinstruction following which involves both creative and fact seeking tasksusing a single fixed temperature across all examples and tokens. In this workwe introduce Adaptive Decoding a layer added to the model to select thesampling temperature dynamically at inference time at either the token orexample level in order to optimize performance. To learn its parameters weintroduce Latent Preference Optimization LPO a general approach to traindiscrete latent variables such as choices of temperature. Our methodoutperforms all fixed decoding temperatures across a range of tasks thatrequire different temperatures including UltraFeedback Creative StoryWriting and GSM8K. |


| Item |Content|
| --- |---|
|idx| 2411.09642v1 |
|title| On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse |
|authors| Alkis KalavasisAnay MehrotraGrigoris Velegkas
|links| http://arxiv.org/abs/2411.09642v1 |
|updated| 2024-11-14 18:06:55 UTC |
|summary| Specifying all desirable properties of a language model is challenging butcertain requirements seem essential. Given samples from an unknown languagethe trained model should produce valid strings not seen in training and beexpressive enough to capture the languages full richness. Otherwiseoutputting invalid strings constitutes hallucination and failing to capturethe full range leads to mode collapse. We ask if a language model can meetboth requirements.  We investigate this within a statistical language generation setting buildingon Gold and Angluin. Here the model receives random samples from adistribution over an unknown language K which belongs to a possibly infinitecollection of languages. The goal is to generate unseen strings from K. We saythe model generates from K with consistency and breadth if as training sizeincreases its output converges to all unseen strings in K.  Kleinberg and Mullainathan KM24 asked if consistency and breadth inlanguage generation are possible. We answer this negatively: for a large classof language models including next-token prediction models this is impossiblefor most collections of candidate languages. This contrasts with KM24sresult showing consistent generation without breadth is possible for anycountable collection of languages. Our finding highlights that generation withbreadth fundamentally differs from generation without breadth.  As a byproduct we establish near-tight bounds on the number of samplesneeded for generation with or without breadth.  Finally our results offer hope: consistent generation with breadth isachievable for any countable collection of languages when negative examplesstrings outside K are available alongside positive ones. This suggests thatpost-training feedback which encodes negative examples can be crucial inreducing hallucinations while limiting mode collapse. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2411.09702v1 |
|title| On the Surprising Effectiveness of Attention Transfer for Vision Transformers |
|authors| Alexander C. LiYuandong TianBeidi ChenDeepak PathakXinlei Chen
|links| http://arxiv.org/abs/2411.09702v1 |
|updated| 2024-11-14 18:59:40 UTC |
|summary| Conventional wisdom suggests that pre-training Vision Transformers ViTimproves downstream performance by learning useful representations. Is thisactually true We investigate this question and find that the features andrepresentations learned during pre-training are not essential. Surprisinglyusing only the attention patterns from pre-training i.e. guiding howinformation flows between tokens is sufficient for models to learn highquality features from scratch and achieve comparable downstream performance. Weshow this by introducing a simple method called attention transfer where onlythe attention patterns from a pre-trained teacher ViT are transferred to astudent either by copying or distilling the attention maps. Since attentiontransfer lets the student learn its own features ensembling it with afine-tuned teacher also further improves accuracy on ImageNet. Wesystematically study various aspects of our findings on the sufficiency ofattention maps including distribution shift settings where they underperformfine-tuning. We hope our exploration provides a better understanding of whatpre-training accomplishes and leads to a useful alternative to the standardpractice of fine-tuning |


| Item |Content|
| --- |---|
|idx| 2411.09689v1 |
|title| LLM Hallucination Reasoning with Zero-shot Knowledge Test |
|authors| Seongmin LeeHsiang HsuChun-Fu Chen
|links| http://arxiv.org/abs/2411.09689v1 |
|updated| 2024-11-14 18:55:26 UTC |
|summary| LLM hallucination where LLMs occasionally generate unfaithful text posessignificant challenges for their practical applications. Most existingdetection methods rely on external knowledge LLM fine-tuning orhallucination-labeled datasets and they do not distinguish between differenttypes of hallucinations which are crucial for improving detection performance.We introduce a new task Hallucination Reasoning which classifiesLLM-generated text into one of three categories: aligned misaligned andfabricated. Our novel zero-shot method assesses whether LLM has enoughknowledge about a given prompt and text. Our experiments conducted on newdatasets demonstrate the effectiveness of our method in hallucination reasoningand underscore its importance for enhancing detection performance. |


| Item |Content|
| --- |---|
|idx| 2411.09683v1 |
|title| Towards a Classification of Open-Source ML Models and Datasets for Software Engineering |
|authors| Alexandra GonzálezXavier FranchDavid LoSilverio Martínez-Fernández
|links| http://arxiv.org/abs/2411.09683v1 |
|updated| 2024-11-14 18:52:05 UTC |
|summary| Background: Open-Source Pre-Trained Models PTMs and datasets provideextensive resources for various Machine Learning ML tasks yet theseresources lack a classification tailored to Software Engineering SE needs.Aims: We apply an SE-oriented classification to PTMs and datasets on a popularopen-source ML repository Hugging Face HF and analyze the evolution of PTMsover time. Method: We conducted a repository mining study. We started with asystematically gathered database of PTMs and datasets from the HF API. Ourselection was refined by analyzing model and dataset cards and metadata suchas tags and confirming SE relevance using Gemini 1.5 Pro. All analyses arereplicable with a publicly accessible replication package. Results: The mostcommon SE task among PTMs and datasets is code generation with a primary focuson software development and limited attention to software management. PopularPTMs and datasets mainly target software development. Among ML tasks textgeneration is the most common in SE PTMs and datasets. There has been a markedincrease in PTMs for SE since 2023 Q2. Conclusions: This study underscores theneed for broader task coverage to enhance the integration of ML within SEpractices. |


| Item |Content|
| --- |---|
|idx| 2411.09678v1 |
|title| NeuralDEM -- Real-time Simulation of Industrial Particulate Flows |
|authors| Benedikt AlkinTobias KronlachnerSamuele PapaStefan PirkerThomas LichteneggerJohannes Brandstetter
|links| http://arxiv.org/abs/2411.09678v1 |
|updated| 2024-11-14 18:44:31 UTC |
|summary| Advancements in computing power have made it possible to numerically simulatelarge-scale fluid-mechanical and/or particulate systems many of which areintegral to core industrial processes. Among the different numerical methodsavailable the discrete element method DEM provides one of the most accuraterepresentations of a wide range of physical systems involving granular anddiscontinuous materials. Consequently DEM has become a widely acceptedapproach for tackling engineering problems connected to granular flows andpowder mechanics. Additionally DEM can be integrated with grid-basedcomputational fluid dynamics CFD methods enabling the simulation of chemicalprocesses taking place e.g. in fluidized beds. However DEM iscomputationally intensive because of the intrinsic multiscale nature ofparticulate systems restricting simulation duration or number of particles.Towards this end NeuralDEM presents an end-to-end approach to replace slownumerical DEM routines with fast adaptable deep learning surrogates. NeuralDEMis capable of picturing long-term transport processes across different regimesusing macroscopic observables without any reference to microscopic modelparameters. First NeuralDEM treats the Lagrangian discretization of DEM as anunderlying continuous field while simultaneously modeling macroscopic behaviordirectly as additional auxiliary fields. Second NeuralDEM introducesmulti-branch neural operators scalable to real-time modeling ofindustrially-sized scenarios - from slow and pseudo-steady to fast andtransient. Such scenarios have previously posed insurmountable challenges fordeep learning models. Notably NeuralDEM faithfully models coupled CFD-DEMfluidized bed reactors of 160k CFD cells and 500k DEM particles fortrajectories of 28s. NeuralDEM will open many new doors to advanced engineeringand much faster process cycles. |


| Item |Content|
| --- |---|
|idx| 2411.09648v1 |
|title| Med-Bot: An AI-Powered Assistant to Provide Accurate and Reliable Medical Information |
|authors| Ahan BhattNandan Vaghela
|links| http://arxiv.org/abs/2411.09648v1 |
|updated| 2024-11-14 18:17:30 UTC |
|summary| This paper introduces Med-Bot an AI-powered chatbot designed to provideusers with accurate and reliable medical information. Utilizing advancedlibraries and frameworks such as PyTorch Chromadb Langchain and AutogptqMed-Bot is built to handle the complexities of natural language understandingin a healthcare context. The integration of llamaassisted data processing andAutoGPT-Q provides enhanced performance in processing and responding to queriesbased on PDFs of medical literature ensuring that users receive precise andtrustworthy information. This research details the methodologies employed indeveloping Med-Bot and evaluates its effectiveness in disseminating healthcareinformation. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2411.09702v1 |
|title| On the Surprising Effectiveness of Attention Transfer for Vision Transformers |
|authors| Alexander C. LiYuandong TianBeidi ChenDeepak PathakXinlei Chen
|links| http://arxiv.org/abs/2411.09702v1 |
|updated| 2024-11-14 18:59:40 UTC |
|summary| Conventional wisdom suggests that pre-training Vision Transformers ViTimproves downstream performance by learning useful representations. Is thisactually true We investigate this question and find that the features andrepresentations learned during pre-training are not essential. Surprisinglyusing only the attention patterns from pre-training i.e. guiding howinformation flows between tokens is sufficient for models to learn highquality features from scratch and achieve comparable downstream performance. Weshow this by introducing a simple method called attention transfer where onlythe attention patterns from a pre-trained teacher ViT are transferred to astudent either by copying or distilling the attention maps. Since attentiontransfer lets the student learn its own features ensembling it with afine-tuned teacher also further improves accuracy on ImageNet. Wesystematically study various aspects of our findings on the sufficiency ofattention maps including distribution shift settings where they underperformfine-tuning. We hope our exploration provides a better understanding of whatpre-training accomplishes and leads to a useful alternative to the standardpractice of fine-tuning |


| Item |Content|
| --- |---|
|idx| 2411.09686v1 |
|title| Conditional regression for the Nonlinear Single-Variable Model |
|authors| Yantao WuMauro Maggioni
|links| http://arxiv.org/abs/2411.09686v1 |
|updated| 2024-11-14 18:53:51 UTC |
|summary| Several statistical models for regression of a function F on mathbbRdwithout the statistical and computational curse of dimensionality exist forexample by imposing and exploiting geometric assumptions on the distribution ofthe data e.g. that its support is low-dimensional or strong smoothnessassumptions on F or a special structure F. Among the latter compositionalmodels assume Ffcirc g with g mapping to mathbbRr with rll dhave been studied and include classical single- and multi-index models andrecent works on neural networks. While the case where g is linear is ratherwell-understood much less is known when g is nonlinear and in particularfor which gs the curse of dimensionality in estimating F or both f andg may be circumvented. In this paper we consider a modelFX:fPi_gamma X  where Pi_gamma:mathbbRdto0rmlen_gammais the closest-point projection onto the parameter of a regular curve gamma:0rmlen_gammatomathbbRd and f:0rmlen_gammatomathbbR1.The input data X is not low-dimensional far from gamma conditioned onPi_gammaX being well-defined. The distribution of the data gamma andf are unknown. This model is a natural nonlinear generalization of thesingle-index model which corresponds to gamma being a line. We propose anonparametric estimator based on conditional regression and show that undersuitable assumptions the strongest of which being that f is coarselymonotone it can achieve the one-dimensional optimal min-max rate fornon-parametric regression up to the level of noise in the observations and beconstructed in time mathcalOd2nlog n. All the constants in thelearning bounds in the minimal number of samples required for our bounds tohold and in the computational complexity are at most low-order polynomials ind. |


| Item |Content|
| --- |---|
|idx| 2411.09683v1 |
|title| Towards a Classification of Open-Source ML Models and Datasets for Software Engineering |
|authors| Alexandra GonzálezXavier FranchDavid LoSilverio Martínez-Fernández
|links| http://arxiv.org/abs/2411.09683v1 |
|updated| 2024-11-14 18:52:05 UTC |
|summary| Background: Open-Source Pre-Trained Models PTMs and datasets provideextensive resources for various Machine Learning ML tasks yet theseresources lack a classification tailored to Software Engineering SE needs.Aims: We apply an SE-oriented classification to PTMs and datasets on a popularopen-source ML repository Hugging Face HF and analyze the evolution of PTMsover time. Method: We conducted a repository mining study. We started with asystematically gathered database of PTMs and datasets from the HF API. Ourselection was refined by analyzing model and dataset cards and metadata suchas tags and confirming SE relevance using Gemini 1.5 Pro. All analyses arereplicable with a publicly accessible replication package. Results: The mostcommon SE task among PTMs and datasets is code generation with a primary focuson software development and limited attention to software management. PopularPTMs and datasets mainly target software development. Among ML tasks textgeneration is the most common in SE PTMs and datasets. There has been a markedincrease in PTMs for SE since 2023 Q2. Conclusions: This study underscores theneed for broader task coverage to enhance the integration of ML within SEpractices. |


| Item |Content|
| --- |---|
|idx| 2411.09678v1 |
|title| NeuralDEM -- Real-time Simulation of Industrial Particulate Flows |
|authors| Benedikt AlkinTobias KronlachnerSamuele PapaStefan PirkerThomas LichteneggerJohannes Brandstetter
|links| http://arxiv.org/abs/2411.09678v1 |
|updated| 2024-11-14 18:44:31 UTC |
|summary| Advancements in computing power have made it possible to numerically simulatelarge-scale fluid-mechanical and/or particulate systems many of which areintegral to core industrial processes. Among the different numerical methodsavailable the discrete element method DEM provides one of the most accuraterepresentations of a wide range of physical systems involving granular anddiscontinuous materials. Consequently DEM has become a widely acceptedapproach for tackling engineering problems connected to granular flows andpowder mechanics. Additionally DEM can be integrated with grid-basedcomputational fluid dynamics CFD methods enabling the simulation of chemicalprocesses taking place e.g. in fluidized beds. However DEM iscomputationally intensive because of the intrinsic multiscale nature ofparticulate systems restricting simulation duration or number of particles.Towards this end NeuralDEM presents an end-to-end approach to replace slownumerical DEM routines with fast adaptable deep learning surrogates. NeuralDEMis capable of picturing long-term transport processes across different regimesusing macroscopic observables without any reference to microscopic modelparameters. First NeuralDEM treats the Lagrangian discretization of DEM as anunderlying continuous field while simultaneously modeling macroscopic behaviordirectly as additional auxiliary fields. Second NeuralDEM introducesmulti-branch neural operators scalable to real-time modeling ofindustrially-sized scenarios - from slow and pseudo-steady to fast andtransient. Such scenarios have previously posed insurmountable challenges fordeep learning models. Notably NeuralDEM faithfully models coupled CFD-DEMfluidized bed reactors of 160k CFD cells and 500k DEM particles fortrajectories of 28s. NeuralDEM will open many new doors to advanced engineeringand much faster process cycles. |


| Item |Content|
| --- |---|
|idx| 2411.09648v1 |
|title| Med-Bot: An AI-Powered Assistant to Provide Accurate and Reliable Medical Information |
|authors| Ahan BhattNandan Vaghela
|links| http://arxiv.org/abs/2411.09648v1 |
|updated| 2024-11-14 18:17:30 UTC |
|summary| This paper introduces Med-Bot an AI-powered chatbot designed to provideusers with accurate and reliable medical information. Utilizing advancedlibraries and frameworks such as PyTorch Chromadb Langchain and AutogptqMed-Bot is built to handle the complexities of natural language understandingin a healthcare context. The integration of llamaassisted data processing andAutoGPT-Q provides enhanced performance in processing and responding to queriesbased on PDFs of medical literature ensuring that users receive precise andtrustworthy information. This research details the methodologies employed indeveloping Med-Bot and evaluates its effectiveness in disseminating healthcareinformation. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2411.09703v1 |
|title| MagicQuill: An Intelligent Interactive Image Editing System |
|authors| Zichen LiuYue YuHao OuyangQiuyu WangKa Leong ChengWen WangZhiheng LiuQifeng ChenYujun Shen
|links| http://arxiv.org/abs/2411.09703v1 |
|updated| 2024-11-14 18:59:57 UTC |
|summary| Image editing involves a variety of complex tasks and requires efficient andprecise manipulation techniques. In this paper we present MagicQuill anintegrated image editing system that enables swift actualization of creativeideas. Our system features a streamlined yet functionally robust interfaceallowing for the articulation of editing operations e.g. inserting elementserasing objects altering color with minimal input. These interactions aremonitored by a multimodal large language model MLLM to anticipate editingintentions in real time bypassing the need for explicit prompt entry. Finallywe apply a powerful diffusion prior enhanced by a carefully learned two-branchplug-in module to process editing requests with precise control. Experimentalresults demonstrate the effectiveness of MagicQuill in achieving high-qualityimage edits. Please visit https://magic-quill.github.io to try out our system. |


| Item |Content|
| --- |---|
|idx| 2411.09702v1 |
|title| On the Surprising Effectiveness of Attention Transfer for Vision Transformers |
|authors| Alexander C. LiYuandong TianBeidi ChenDeepak PathakXinlei Chen
|links| http://arxiv.org/abs/2411.09702v1 |
|updated| 2024-11-14 18:59:40 UTC |
|summary| Conventional wisdom suggests that pre-training Vision Transformers ViTimproves downstream performance by learning useful representations. Is thisactually true We investigate this question and find that the features andrepresentations learned during pre-training are not essential. Surprisinglyusing only the attention patterns from pre-training i.e. guiding howinformation flows between tokens is sufficient for models to learn highquality features from scratch and achieve comparable downstream performance. Weshow this by introducing a simple method called attention transfer where onlythe attention patterns from a pre-trained teacher ViT are transferred to astudent either by copying or distilling the attention maps. Since attentiontransfer lets the student learn its own features ensembling it with afine-tuned teacher also further improves accuracy on ImageNet. Wesystematically study various aspects of our findings on the sufficiency ofattention maps including distribution shift settings where they underperformfine-tuning. We hope our exploration provides a better understanding of whatpre-training accomplishes and leads to a useful alternative to the standardpractice of fine-tuning |


| Item |Content|
| --- |---|
|idx| 2411.09693v1 |
|title| CropCraft: Inverse Procedural Modeling for 3D Reconstruction of Crop Plants |
|authors| Albert J. ZhaiXinlei WangKaiyuan LiZhao JiangJunxiong ZhouSheng WangZhenong JinKaiyu GuanShenlong Wang
|links| http://arxiv.org/abs/2411.09693v1 |
|updated| 2024-11-14 18:58:02 UTC |
|summary| The ability to automatically build 3D digital twins of plants from images hascountless applications in agriculture environmental science robotics andother fields. However current 3D reconstruction methods fail to recovercomplete shapes of plants due to heavy occlusion and complex geometries. Inthis work we present a novel method for 3D reconstruction of agriculturalcrops based on optimizing a parametric model of plant morphology via inverseprocedural modeling. Our method first estimates depth maps by fitting a neuralradiance field and then employs Bayesian optimization to estimate plantmorphological parameters that result in consistent depth renderings. Theresulting 3D model is complete and biologically plausible. We validate ourmethod on a dataset of real images of agricultural fields and demonstrate thatthe reconstructions can be used for a variety of monitoring and simulationapplications. |


| Item |Content|
| --- |---|
|idx| 2411.09691v1 |
|title| Advancing Fine-Grained Visual Understanding with Multi-Scale Alignment in Multi-Modal Models |
|authors| Wei WangZhaowei LiQi XuLinfeng LiYiQing CaiBotian JiangHang SongXingcan HuPengyu WangLi Xiao
|links| http://arxiv.org/abs/2411.09691v1 |
|updated| 2024-11-14 18:57:07 UTC |
|summary| Multi-modal large language models MLLMs have achieved remarkable success infine-grained visual understanding across a range of tasks. However they oftenencounter significant challenges due to inadequate alignment for fine-grainedknowledge which restricts their ability to accurately capture local detailsand attain a comprehensive global perception. While recent advancements havefocused on aligning object expressions with grounding information theytypically lack explicit integration of object images which contain affluentinformation beyond mere texts or coordinates. To bridge this gap we introducea novel fine-grained visual knowledge alignment method that effectively alignsand integrates multi-scale knowledge of objects including texts coordinatesand images. This innovative method is underpinned by our multi-scalefine-grained enhancement data synthesis pipeline which provides over 300Kessential training data to enhance alignment and improve overall performance.Furthermore we present TinyGroundingGPT a series of compact models optimizedfor high-level alignments. With a scale of approximately 3B parametersTinyGroundingGPT achieves outstanding results in grounding tasks whiledelivering performance comparable to larger MLLMs in complex visual scenarios. |


| Item |Content|
| --- |---|
|idx| 2411.09627v1 |
|title| One-Shot Manipulation Strategy Learning by Making Contact Analogies |
|authors| Yuyao LiuJiayuan MaoJoshua TenenbaumTomás Lozano-PérezLeslie Pack Kaelbling
|links| http://arxiv.org/abs/2411.09627v1 |
|updated| 2024-11-14 17:54:43 UTC |
|summary| We present a novel approach MAGIC manipulation analogies for generalizableintelligent contacts for one-shot learning of manipulation strategies withfast and extensive generalization to novel objects. By leveraging a referenceaction trajectory MAGIC effectively identifies similar contact points andsequences of actions on novel objects to replicate a demonstrated strategysuch as using different hooks to retrieve distant objects of different shapesand sizes. Our method is based on a two-stage contact-point matching processthat combines global shape matching using pretrained neural features with localcurvature analysis to ensure precise and physically plausible contact points.We experiment with three tasks including scooping hanging and hookingobjects. MAGIC demonstrates superior performance over existing methodsachieving significant improvements in runtime speed and generalization todifferent object categories. Website: https://magic-2024.github.io/ . |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2411.09686v1 |
|title| Conditional regression for the Nonlinear Single-Variable Model |
|authors| Yantao WuMauro Maggioni
|links| http://arxiv.org/abs/2411.09686v1 |
|updated| 2024-11-14 18:53:51 UTC |
|summary| Several statistical models for regression of a function F on mathbbRdwithout the statistical and computational curse of dimensionality exist forexample by imposing and exploiting geometric assumptions on the distribution ofthe data e.g. that its support is low-dimensional or strong smoothnessassumptions on F or a special structure F. Among the latter compositionalmodels assume Ffcirc g with g mapping to mathbbRr with rll dhave been studied and include classical single- and multi-index models andrecent works on neural networks. While the case where g is linear is ratherwell-understood much less is known when g is nonlinear and in particularfor which gs the curse of dimensionality in estimating F or both f andg may be circumvented. In this paper we consider a modelFX:fPi_gamma X  where Pi_gamma:mathbbRdto0rmlen_gammais the closest-point projection onto the parameter of a regular curve gamma:0rmlen_gammatomathbbRd and f:0rmlen_gammatomathbbR1.The input data X is not low-dimensional far from gamma conditioned onPi_gammaX being well-defined. The distribution of the data gamma andf are unknown. This model is a natural nonlinear generalization of thesingle-index model which corresponds to gamma being a line. We propose anonparametric estimator based on conditional regression and show that undersuitable assumptions the strongest of which being that f is coarselymonotone it can achieve the one-dimensional optimal min-max rate fornon-parametric regression up to the level of noise in the observations and beconstructed in time mathcalOd2nlog n. All the constants in thelearning bounds in the minimal number of samples required for our bounds tohold and in the computational complexity are at most low-order polynomials ind. |


| Item |Content|
| --- |---|
|idx| 2411.09642v1 |
|title| On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse |
|authors| Alkis KalavasisAnay MehrotraGrigoris Velegkas
|links| http://arxiv.org/abs/2411.09642v1 |
|updated| 2024-11-14 18:06:55 UTC |
|summary| Specifying all desirable properties of a language model is challenging butcertain requirements seem essential. Given samples from an unknown languagethe trained model should produce valid strings not seen in training and beexpressive enough to capture the languages full richness. Otherwiseoutputting invalid strings constitutes hallucination and failing to capturethe full range leads to mode collapse. We ask if a language model can meetboth requirements.  We investigate this within a statistical language generation setting buildingon Gold and Angluin. Here the model receives random samples from adistribution over an unknown language K which belongs to a possibly infinitecollection of languages. The goal is to generate unseen strings from K. We saythe model generates from K with consistency and breadth if as training sizeincreases its output converges to all unseen strings in K.  Kleinberg and Mullainathan KM24 asked if consistency and breadth inlanguage generation are possible. We answer this negatively: for a large classof language models including next-token prediction models this is impossiblefor most collections of candidate languages. This contrasts with KM24sresult showing consistent generation without breadth is possible for anycountable collection of languages. Our finding highlights that generation withbreadth fundamentally differs from generation without breadth.  As a byproduct we establish near-tight bounds on the number of samplesneeded for generation with or without breadth.  Finally our results offer hope: consistent generation with breadth isachievable for any countable collection of languages when negative examplesstrings outside K are available alongside positive ones. This suggests thatpost-training feedback which encodes negative examples can be crucial inreducing hallucinations while limiting mode collapse. |


| Item |Content|
| --- |---|
|idx| 2411.09635v1 |
|title| Counterfactual Uncertainty Quantification of Factual Estimand of Efficacy from Before-and-After Treatment Repeated Measures Randomized Controlled Trials |
|authors| Xingya WangYang HanYushi LiuSzu-Yu TangJason C. Hsu
|links| http://arxiv.org/abs/2411.09635v1 |
|updated| 2024-11-14 18:01:02 UTC |
|summary| The ideal estimand for comparing a new treatment Rx with a control C isthe textitcounterfactual efficacy Rx:C the expected differentialoutcome between Rx and C if each patient were given textitboth. Whilecounterfactual textitpoint estimation from textitfactual RandomizedControlled Trials RCTs has been available this article showstextitcounterfactual uncertainty quantification CUQ quantifyinguncertainty for factual point estimates but in a counterfactual setting issurprisingly achievable. We achieve CUQ whose variability is typically smallerthan factual UQ by creating a new statistical modeling principle called ETZwhich is applicable to RCTs with textitBefore-and-After treatment RepeatedMeasures common in many therapeutic areas.  We urge caution when estimate of the unobservable true condition of a patientbefore treatment has measurement error because that violation of standardregression assumption can cause attenuation in estimating treatment effects.Fortunately we prove that for traditional medicine in general and fortargeted therapy with efficacy defined as averaged over the populationcounterfactual point estimation is unbiased. However for targeted therapyboth Real Human and Digital Twins approaches should respect this limitationlest predicted treatment effect in textitsubgroups will have bias. |


| Item |Content|
| --- |---|
|idx| 2411.09516v1 |
|title| Sharp Matrix Empirical Bernstein Inequalities |
|authors| Hongjian WangAaditya Ramdas
|links| http://arxiv.org/abs/2411.09516v1 |
|updated| 2024-11-14 15:27:18 UTC |
|summary| We present two sharp empirical Bernstein inequalities for symmetric randommatrices with bounded eigenvalues. By sharp we mean that both inequalitiesadapt to the unknown variance in a tight manner: the deviation captured by thefirst-order 1/sqrtn term asymptotically matches the matrix Bernsteininequality exactly including constants the latter requiring knowledge of thevariance. Our first inequality holds for the sample mean of independentmatrices and our second inequality holds for a mean estimator under martingaledependence at stopping times. |


| Item |Content|
| --- |---|
|idx| 2411.09483v1 |
|title| Sparse Bayesian Generative Modeling for Compressive Sensing |
|authors| Benedikt BöckSadaf SyedWolfgang Utschick
|links| http://arxiv.org/abs/2411.09483v1 |
|updated| 2024-11-14 14:37:47 UTC |
|summary| This work addresses the fundamental linear inverse problem in compressivesensing CS by introducing a new type of regularizing generative prior. Ourproposed method utilizes ideas from classical dictionary-based CS and inparticular sparse Bayesian learning SBL to integrate a strongregularization towards sparse solutions. At the same time by leveraging thenotion of conditional Gaussianity it also incorporates the adaptability fromgenerative models to training data. However unlike most state-of-the-artgenerative models it is able to learn from a few compressed and noisy datasamples and requires no optimization algorithm for solving the inverse problem.Additionally similar to Dirichlet prior networks our model parameterizes aconjugate prior enabling its application for uncertainty quantification. Wesupport our approach theoretically through the concept of variational inferenceand validate it empirically using different types of compressible signals. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2411.09577v1 |
|title| SimTube: Generating Simulated Video Comments through Multimodal AI and User Personas |
|authors| Yu-Kai HungYun-Chien HuangTing-Yu SuYen-Ting LinLung-Pan ChengBryan WangShao-Hua Sun
|links| http://arxiv.org/abs/2411.09577v1 |
|updated| 2024-11-14 16:35:17 UTC |
|summary| Audience feedback is crucial for refining video content yet it typicallycomes after publication limiting creators ability to make timely adjustments.To bridge this gap we introduce SimTube a generative AI system designed tosimulate audience feedback in the form of video comments before a videosrelease. SimTube features a computational pipeline that integrates multimodaldata from the video-such as visuals audio and metadata-with user personasderived from a broad and diverse corpus of audience demographics generatingvaried and contextually relevant feedback. Furthermore the systems UI allowscreators to explore and customize the simulated comments. Through acomprehensive evaluation-comprising quantitative analysis crowd-sourcedassessments and qualitative user studies-we show that SimTubes generatedcomments are not only relevant believable and diverse but often more detailedand informative than actual audience comments highlighting its potential tohelp creators refine their content before release. |


| Item |Content|
| --- |---|
|idx| 2411.09436v1 |
|title| Robot Tasks with Fuzzy Time Requirements from Natural Language Instructions |
|authors| Sascha SuckerMichael NeubauerDominik Henrich
|links| http://arxiv.org/abs/2411.09436v1 |
|updated| 2024-11-14 13:34:16 UTC |
|summary| Natural language allows robot programming to be accessible to everyone.However the inherent fuzziness in natural language poses challenges forinflexible traditional robot systems. We focus on instructions with fuzzy timerequirements e.g. start in a few minutes. Building on previous roboticsresearch we introduce fuzzy skills. These define an execution by the robotwith so-called satisfaction functions representing vague execution timerequirements. Such functions express a users satisfaction over potentialstarting times for skill execution. When the robot handles multiple fuzzyskills the satisfaction function provides a temporal tolerance window forexecution thus enabling optimal scheduling based on satisfaction. Wegeneralized such functions based on individual user expectations with a userstudy. The participants rated their satisfaction with an instructionsexecution at various times. Our investigations reveal that trapezoidalfunctions best approximate the users satisfaction. Additionally the resultssuggest that users are more lenient if the execution is specified further intothe future. |


| Item |Content|
| --- |---|
|idx| 2411.09266v1 |
|title| How Good is ChatGPT at Audiovisual Deepfake Detection: A Comparative Study of ChatGPT, AI Models and Human Perception |
|authors| Sahibzada Adil ShahzadAmmarah HashmiYan-Tsung PengYu TsaoHsin-Min Wang
|links| http://arxiv.org/abs/2411.09266v1 |
|updated| 2024-11-14 08:07:02 UTC |
|summary| Multimodal deepfakes involving audiovisual manipulations are a growing threatbecause they are difficult to detect with the naked eye or using unimodal deeplearningbased forgery detection methods. Audiovisual forensic models whilemore capable than unimodal models require large training datasets and arecomputationally expensive for training and inference. Furthermore these modelslack interpretability and often do not generalize well to unseen manipulations.In this study we examine the detection capabilities of a large language modelLLM i.e. ChatGPT to identify and account for any possible visual andauditory artifacts and manipulations in audiovisual deepfake content. Extensiveexperiments are conducted on videos from a benchmark multimodal deepfakedataset to evaluate the detection performance of ChatGPT and compare it withthe detection capabilities of state-of-the-art multimodal forensic models andhumans. Experimental results demonstrate the importance of domain knowledge andprompt engineering for video forgery detection tasks using LLMs. Unlikeapproaches based on end-to-end learning ChatGPT can account for spatial andspatiotemporal artifacts and inconsistencies that may exist within or acrossmodalities. Additionally we discuss the limitations of ChatGPT for multimediaforensic tasks. |


| Item |Content|
| --- |---|
|idx| 2411.09169v1 |
|title| Artificial Theory of Mind and Self-Guided Social Organisation |
|authors| Michael S. HarréJaime Ruiz-SerraCatherine Drysdale
|links| http://arxiv.org/abs/2411.09169v1 |
|updated| 2024-11-14 04:06:26 UTC |
|summary| One of the challenges artificial intelligence AI faces is how a collectionof agents coordinate their behaviour to achieve goals that are not reachable byany single agent. In a recent article by Ozmen et al this was framed as one ofsix grand challenges: That AI needs to respect human cognitive processes at thehuman-AI interaction frontier. We suggest that this extends to the AI-AIfrontier and that it should also reflect human psychology as it is the onlysuccessful framework we have from which to build out. In this extended abstractwe first make the case for collective intelligence in a general settingdrawing on recent work from single neuron complexity in neural networks and antnetwork adaptability in ant colonies. From there we introduce how speciesrelate to one another in an ecological network via niche selection nichechoice and niche conformity with the aim of forming an analogy with humansocial network development as new agents join together and coordinate. Fromthere we show how our social structures are influenced by our neuro-physiologyour psychology and our language. This emphasises how individual people withina social network influence the structure and performance of that network incomplex tasks and that cognitive faculties such as Theory of Mind play acentral role. We finish by discussing the current state of the art in AI andwhere there is potential for further development of a socially embodiedcollective artificial intelligence that is capable of guiding its own socialstructures. |


| Item |Content|
| --- |---|
|idx| 2411.09102v1 |
|title| Provocation: Who benefits from "inclusion" in Generative AI? |
|authors| Nari JohnsonSiobhan Mackenzie HallSamantha Dalal
|links| http://arxiv.org/abs/2411.09102v1 |
|updated| 2024-11-14 00:18:25 UTC |
|summary| The demands for accurate and representative generative AI systems means thereis an increased demand on participatory evaluation structures. While theseparticipatory structures are paramount to to ensure non-dominant valuesknowledge and material culture are also reflected in AI models and the mediathey generate we argue that dominant structures of community participation inAI development and evaluation are not explicit enough about the benefits andharms that members of socially marginalized groups may experience as a resultof their participation. Without explicit interrogation of these benefits by AIdevelopers as a community we may remain blind to the immensity of systemicchange that is needed as well. To support this provocation we present aspeculative case study developed from our own collective experiences as AIresearchers. We use this speculative context to itemize the barriers that needto be overcome in order for the proposed benefits to marginalized communitiesto be realized and harms mitigated. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2411.09636v1 |
|title| Nash equilibrium seeking for a class of quadratic-bilinear Wasserstein distributionally robust games |
|authors| Georgios PantazisReza Rahimi BahbadoraniSergio Grammatico
|links| http://arxiv.org/abs/2411.09636v1 |
|updated| 2024-11-14 18:03:12 UTC |
|summary| We consider a class of Wasserstein distributionally robust Nash equilibriumproblems where agents construct heterogeneous data-driven Wassersteinambiguity sets using private samples and radii in line with their individualrisk-averse behaviour. By leveraging relevant properties of this class ofgames we show that equilibria of the original seemingly infinite-dimensionalproblem can be obtained as a solution to a finite-dimensional Nash equilibriumproblem. We then reformulate the problem as a finite-dimensional variationalinequality and establish the connection between the corresponding solutionsets. Our reformulation has scalable behaviour with respect to the data sizeand maintains a fixed number of constraints independently of the number ofsamples. To compute a solution we leverage two algorithms based on the goldenratio algorithm. The efficiency of both algorithmic schemes is corroboratedthrough extensive simulation studies on an illustrative example and astochastic portfolio allocation game where behavioural coupling amonginvestors is modeled. |


| Item |Content|
| --- |---|
|idx| 2411.09493v1 |
|title| Strategic Sacrifice: Self-Organized Robot Swarm Localization for Inspection Productivity |
|authors| Sneha RamshankerHungtang KoRadhika Nagpal
|links| http://arxiv.org/abs/2411.09493v1 |
|updated| 2024-11-14 15:00:14 UTC |
|summary| Robot swarms offer significant potential for inspecting diverseinfrastructure ranging from bridges to space stations. However effectiveinspection requires accurate robot localization which demands substantialcomputational resources and limits productivity. Inspired by biologicalsystems we introduce a novel cooperative localization mechanism that minimizescollective computation expenditure through self-organized sacrifice. Here afew agents bear the computational burden of localization through localinteractions they improve the inspection productivity of the swarm. Ourapproach adaptively maximizes inspection productivity for unconstrainedtrajectories in dynamic interaction and environmental settings. We demonstratethe optimality and robustness using mean-field analytical models multi-agentsimulations and hardware experiments with metal climbing robots inspecting a3D cylinder. |


| Item |Content|
| --- |---|
|idx| 2411.09191v1 |
|title| Informational Puts |
|authors| Andrew KohSivakorn SanguanmooKei Uzui
|links| http://arxiv.org/abs/2411.09191v1 |
|updated| 2024-11-14 05:10:48 UTC |
|summary| We fully characterize how dynamic information should be provided to uniquelyimplement the largest equilibrium in dynamic binary-action supermodular games.The designer offers an informational put: she stays silent in good times butinjects asymmetric and inconclusive public information if players lose faith.There is i no multiplicity gap: the largest partially implementableequilibrium can be implemented uniquely and ii no intertemporal commitmentgap: the policy is sequentially optimal. Our results have sharp implicationsfor the design of policy in coordination environments. |


| Item |Content|
| --- |---|
|idx| 2411.09169v1 |
|title| Artificial Theory of Mind and Self-Guided Social Organisation |
|authors| Michael S. HarréJaime Ruiz-SerraCatherine Drysdale
|links| http://arxiv.org/abs/2411.09169v1 |
|updated| 2024-11-14 04:06:26 UTC |
|summary| One of the challenges artificial intelligence AI faces is how a collectionof agents coordinate their behaviour to achieve goals that are not reachable byany single agent. In a recent article by Ozmen et al this was framed as one ofsix grand challenges: That AI needs to respect human cognitive processes at thehuman-AI interaction frontier. We suggest that this extends to the AI-AIfrontier and that it should also reflect human psychology as it is the onlysuccessful framework we have from which to build out. In this extended abstractwe first make the case for collective intelligence in a general settingdrawing on recent work from single neuron complexity in neural networks and antnetwork adaptability in ant colonies. From there we introduce how speciesrelate to one another in an ecological network via niche selection nichechoice and niche conformity with the aim of forming an analogy with humansocial network development as new agents join together and coordinate. Fromthere we show how our social structures are influenced by our neuro-physiologyour psychology and our language. This emphasises how individual people withina social network influence the structure and performance of that network incomplex tasks and that cognitive faculties such as Theory of Mind play acentral role. We finish by discussing the current state of the art in AI andwhere there is potential for further development of a socially embodiedcollective artificial intelligence that is capable of guiding its own socialstructures. |


| Item |Content|
| --- |---|
|idx| 2411.09168v1 |
|title| Theory of Mind Enhances Collective Intelligence |
|authors| Michael S. HarréCatherine DrysdaleJaime Ruiz-Serra
|links| http://arxiv.org/abs/2411.09168v1 |
|updated| 2024-11-14 03:58:50 UTC |
|summary| Collective Intelligence plays a central role in a large variety of fieldsfrom economics and evolutionary theory to neural networks and eusocial insectsand it is also core to much of the work on emergence and self-organisation incomplex systems theory. However in human collective intelligence there isstill much more to be understood in the relationship between specificpsychological processes at the individual level and the emergence ofself-organised structures at the social level. Previously psychological factorshave played a relatively minor role in the study of collective intelligence asthe principles are often quite general and applicable to humans just as readilyas insects or other agents without sophisticated psychologies. In this articlewe emphasise with examples from other complex adaptive systems the broadapplicability of collective intelligence principles while the mechanisms andtime-scales differ significantly between examples. We contend that flexiblecollective intelligence in human social settings is improved by our use of aspecific cognitive tool: our Theory of Mind. We identify several keycharacteristics of psychologically mediated collective intelligence and showthat the development of a Theory of Mind is a crucial factor distinguishingsocial collective intelligence from general collective intelligence. We thenplace these capabilities in the context of the next steps in artificialintelligence embedded in a future that includes an effective human-AI hybridsocial ecology. |


