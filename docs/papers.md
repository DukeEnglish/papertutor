# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.19753v1 |
|title| DOCCI: Descriptions of Connected and Contrasting Images |
|authors| Yasumasa OnoeSunayana RaneZachary BergerYonatan BittonJaemin ChoRoopal GargAlexander KuZarana ParekhJordi Pont-TusetGarrett TanzerSu WangJason Baldridge
|links| http://arxiv.org/abs/2404.19753v1 |
|updated| 2024-04-30 17:56:24 UTC |
|summary| Vision-language datasets are vital for both text-to-image T2I andimage-to-text I2T research. However current datasets lack descriptions withfine-grained detail that would allow for richer associations to be learned bymodels. To fill the gap we introduce Descriptions of Connected and ContrastingImages DOCCI a dataset with long human-annotated English descriptions for15k images that were taken curated and donated by a single researcher intenton capturing key challenges such as spatial relations counting textrendering world knowledge and more. We instruct human annotators to createcomprehensive descriptions for each image these average 136 words in lengthand are crafted to clearly distinguish each image from those that are relatedor similar. Each description is highly compositional and typically encompassesmultiple challenges. Through both quantitative and qualitative analyses wedemonstrate that DOCCI serves as an effective training resource forimage-to-text generation -- a PaLI 5B model finetuned on DOCCI shows equal orsuperior results compared to highly-performant larger models like LLaVA-1.5 7Band InstructBLIP 7B. Furthermore we show that DOCCI is a useful testbed fortext-to-image generation highlighting the limitations of current text-to-imagemodels in capturing long descriptions and fine details. |


| Item |Content|
| --- |---|
|idx| 2404.19737v1 |
|title| Better & Faster Large Language Models via Multi-token Prediction |
|authors| Fabian GloeckleBadr Youbi IdrissiBaptiste RozièreDavid Lopez-PazGabriel Synnaeve
|links| http://arxiv.org/abs/2404.19737v1 |
|updated| 2024-04-30 17:33:57 UTC |
|summary| Large language models such as GPT and Llama are trained with a next-tokenprediction loss. In this work we suggest that training language models topredict multiple future tokens at once results in higher sample efficiency.More specifically at each position in the training corpus we ask the model topredict the following n tokens using n independent output heads operating ontop of a shared model trunk. Considering multi-token prediction as an auxiliarytraining task we measure improved downstream capabilities with no overhead intraining time for both code and natural language models. The method isincreasingly useful for larger model sizes and keeps its appeal when trainingfor multiple epochs. Gains are especially pronounced on generative benchmarkslike coding where our models consistently outperform strong baselines byseveral percentage points. Our 13B parameter models solves 12  more problemson HumanEval and 17  more on MBPP than comparable next-token models.Experiments on small algorithmic tasks demonstrate that multi-token predictionis favorable for the development of induction heads and algorithmic reasoningcapabilities. As an additional benefit models trained with 4-token predictionare up to 3 times faster at inference even with large batch sizes. |


| Item |Content|
| --- |---|
|idx| 2404.19733v1 |
|title| Iterative Reasoning Preference Optimization |
|authors| Richard Yuanzhe PangWeizhe YuanKyunghyun ChoHe HeSainbayar SukhbaatarJason Weston
|links| http://arxiv.org/abs/2404.19733v1 |
|updated| 2024-04-30 17:28:05 UTC |
|summary| Iterative preference optimization methods have recently been shown to performwell for general instruction tuning tasks but typically make littleimprovement on reasoning tasks Yuan et al. 2024 Chen et al. 2024. In thiswork we develop an iterative approach that optimizes the preference betweencompeting generated Chain-of-Thought CoT candidates by optimizing for winningvs. losing reasoning steps that lead to the correct answer. We train using amodified DPO loss Rafailov et al. 2023 with an additional negativelog-likelihood term which we find to be crucial. We show reasoning improvesacross repeated iterations of this scheme. While only relying on examples inthe training set our approach results in increasing accuracy forLlama-2-70B-Chat from 55.6 to 81.6 on GSM8K and 88.7 with majority votingout of 32 samples from 12.5 to 20.8 on MATH and from 77.8 to 86.7 onARC-Challenge which outperforms other Llama-2-based models not relying onadditionally sourced datasets. |


| Item |Content|
| --- |---|
|idx| 2404.19721v1 |
|title| PANGeA: Procedural Artificial Narrative using Generative AI for Turn-Based Video Games |
|authors| Steph BuongiornoLawrence Jake KlinkertTanishq ChawlaZixin ZhuangCorey Clark
|links| http://arxiv.org/abs/2404.19721v1 |
|updated| 2024-04-30 17:11:54 UTC |
|summary| This research introduces Procedural Artificial Narrative using Generative AIPANGeA a structured approach for leveraging large language models LLMsguided by a game designers high-level criteria to generate narrative contentfor turn-based role-playing video games RPGs. Distinct from priorapplications of LLMs used for video game design PANGeA innovates by not onlygenerating game level data which includes but is not limited to setting keyitems and non-playable characters NPCs but by also fostering dynamicfree-form interactions between the player and the environment that align withthe procedural game narrative. The NPCs generated by PANGeA arepersonality-biased and express traits from the Big 5 Personality Model in theirgenerated responses. PANGeA addresses challenges behind ingesting free-formtext input which can prompt LLM responses beyond the scope of the gamenarrative. A novel validation system that uses the LLMs intelligence evaluatestext input and aligns generated responses with the unfolding narrative. Makingthese interactions possible PANGeA is supported by a server that hosts acustom memory system that supplies context for augmenting generated responsesthus aligning them with the procedural narrative. For its broad applicationthe server has a REST interface enabling any game engine to integrate directlywith PANGeA as well as an LLM interface adaptable with local or private LLMs.PANGeAs ability to foster dynamic narrative generation by aligning responseswith the procedural narrative is demonstrated through an empirical study andablation test of two versions of a demo game. These are a custombrowser-based GPT and a Unity demo. As the results show PANGeA holds potentialto assist game designers in using LLMs to generate narrative-consistent contenteven when provided varied and unpredictable free-form text input. |


| Item |Content|
| --- |---|
|idx| 2404.19714v1 |
|title| ThangDLU at #SMM4H 2024: Encoder-decoder models for classifying text data on social disorders in children and adolescents |
|authors| Hoang-Thang TaAbu Bakar Siddiqur RahmanLotfollah NajjarAlexander Gelbukh
|links| http://arxiv.org/abs/2404.19714v1 |
|updated| 2024-04-30 17:06:20 UTC |
|summary| This paper describes our participation in Task 3 and Task 5 of the SMM4HSocial Media Mining for Health 2024 Workshop explicitly targeting theclassification challenges within tweet data. Task 3 is a multi-classclassification task centered on tweets discussing the impact of outdoorenvironments on symptoms of social anxiety. Task 5 involves a binaryclassification task focusing on tweets reporting medical disorders in children.We applied transfer learning from pre-trained encoder-decoder models such asBART-base and T5-small to identify the labels of a set of given tweets. We alsopresented some data augmentation methods to see their impact on the modelperformance. Finally the systems obtained the best F1 score of 0.627 in Task 3and the best F1 score of 0.841 in Task 5. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.19756v1 |
|title| KAN: Kolmogorov-Arnold Networks |
|authors| Ziming LiuYixuan WangSachin VaidyaFabian RuehleJames HalversonMarin SoljačićThomas Y. HouMax Tegmark
|links| http://arxiv.org/abs/2404.19756v1 |
|updated| 2024-04-30 17:58:29 UTC |
|summary| Inspired by the Kolmogorov-Arnold representation theorem we proposeKolmogorov-Arnold Networks KANs as promising alternatives to Multi-LayerPerceptrons MLPs. While MLPs have fixed activation functions on nodesneurons KANs have learnable activation functions on edges weights.KANs have no linear weights at all -- every weight parameter is replaced by aunivariate function parametrized as a spline. We show that this seeminglysimple change makes KANs outperform MLPs in terms of accuracy andinterpretability. For accuracy much smaller KANs can achieve comparable orbetter accuracy than much larger MLPs in data fitting and PDE solving.Theoretically and empirically KANs possess faster neural scaling laws thanMLPs. For interpretability KANs can be intuitively visualized and can easilyinteract with human users. Through two examples in mathematics and physicsKANs are shown to be useful collaborators helping scientists rediscovermathematical and physical laws. In summary KANs are promising alternatives forMLPs opening opportunities for further improving todays deep learning modelswhich rely heavily on MLPs. |


| Item |Content|
| --- |---|
|idx| 2404.19753v1 |
|title| DOCCI: Descriptions of Connected and Contrasting Images |
|authors| Yasumasa OnoeSunayana RaneZachary BergerYonatan BittonJaemin ChoRoopal GargAlexander KuZarana ParekhJordi Pont-TusetGarrett TanzerSu WangJason Baldridge
|links| http://arxiv.org/abs/2404.19753v1 |
|updated| 2024-04-30 17:56:24 UTC |
|summary| Vision-language datasets are vital for both text-to-image T2I andimage-to-text I2T research. However current datasets lack descriptions withfine-grained detail that would allow for richer associations to be learned bymodels. To fill the gap we introduce Descriptions of Connected and ContrastingImages DOCCI a dataset with long human-annotated English descriptions for15k images that were taken curated and donated by a single researcher intenton capturing key challenges such as spatial relations counting textrendering world knowledge and more. We instruct human annotators to createcomprehensive descriptions for each image these average 136 words in lengthand are crafted to clearly distinguish each image from those that are relatedor similar. Each description is highly compositional and typically encompassesmultiple challenges. Through both quantitative and qualitative analyses wedemonstrate that DOCCI serves as an effective training resource forimage-to-text generation -- a PaLI 5B model finetuned on DOCCI shows equal orsuperior results compared to highly-performant larger models like LLaVA-1.5 7Band InstructBLIP 7B. Furthermore we show that DOCCI is a useful testbed fortext-to-image generation highlighting the limitations of current text-to-imagemodels in capturing long descriptions and fine details. |


| Item |Content|
| --- |---|
|idx| 2404.19748v1 |
|title| Quantifying Nematodes through Images: Datasets, Models, and Baselines of Deep Learning |
|authors| Zhipeng YuanNasamu MusaKatarzyna DybalMatthew BackDaniel LeybournePo Yang
|links| http://arxiv.org/abs/2404.19748v1 |
|updated| 2024-04-30 17:52:31 UTC |
|summary| Every year plant parasitic nematodes one of the major groups of plantpathogens cause a significant loss of crops worldwide. To mitigate crop yieldlosses caused by nematodes an efficient nematode monitoring method isessential for plant and crop disease management. In other respects efficientnematode detection contributes to medical research and drug discovery asnematodes are model organisms. With the rapid development of computertechnology computer vision techniques provide a feasible solution forquantifying nematodes or nematode infections. In this paper we survey andcategorise the studies and available datasets on nematode detection throughdeep-learning models. To stimulate progress in related research this surveypresents the potential state-of-the-art object detection models trainingtechniques optimisation techniques and evaluation metrics for deep learningbeginners. Moreover seven state-of-the-art object detection models arevalidated on three public datasets and the AgriNema dataset for plant parasiticnematodes to construct a baseline for nematode detection. |


| Item |Content|
| --- |---|
|idx| 2404.19744v1 |
|title| PrivComp-KG : Leveraging Knowledge Graph and Large Language Models for Privacy Policy Compliance Verification |
|authors| Leon GarzaLavanya ElluriAnantaa KotalAritran PiplaiDeepti GuptaAnupam Joshi
|links| http://arxiv.org/abs/2404.19744v1 |
|updated| 2024-04-30 17:44:44 UTC |
|summary| Data protection and privacy is becoming increasingly crucial in the digitalera. Numerous companies depend on third-party vendors and service providers tocarry out critical functions within their operations encompassing tasks suchas data handling and storage. However this reliance introduces potentialvulnerabilities as these vendors security measures and practices may notalways align with the standards expected by regulatory bodies. Businesses arerequired often under the penalty of law to ensure compliance with theevolving regulatory rules. Interpreting and implementing these regulations posechallenges due to their complexity. Regulatory documents are extensivedemanding significant effort for interpretation while vendor-drafted privacypolicies often lack the detail required for full legal compliance leading toambiguity. To ensure a concise interpretation of the regulatory requirementsand compliance of organizational privacy policy with said regulations wepropose a Large Language Model LLM and Semantic Web based approach forprivacy compliance. In this paper we develop the novel Privacy PolicyCompliance Verification Knowledge Graph PrivComp-KG. It is designed toefficiently store and retrieve comprehensive information concerning privacypolicies regulatory frameworks and domain-specific knowledge pertaining tothe legal landscape of privacy. Using Retrieval Augmented Generation weidentify the relevant sections in a privacy policy with correspondingregulatory rules. This information about individual privacy policies ispopulated into the PrivComp-KG. Combining this with the domain context andrules the PrivComp-KG can be queried to check for compliance with privacypolicies by each vendor against relevant policy regulations. We demonstrate therelevance of the PrivComp-KG by verifying compliance of privacy policydocuments for various organizations. |


| Item |Content|
| --- |---|
|idx| 2404.19733v1 |
|title| Iterative Reasoning Preference Optimization |
|authors| Richard Yuanzhe PangWeizhe YuanKyunghyun ChoHe HeSainbayar SukhbaatarJason Weston
|links| http://arxiv.org/abs/2404.19733v1 |
|updated| 2024-04-30 17:28:05 UTC |
|summary| Iterative preference optimization methods have recently been shown to performwell for general instruction tuning tasks but typically make littleimprovement on reasoning tasks Yuan et al. 2024 Chen et al. 2024. In thiswork we develop an iterative approach that optimizes the preference betweencompeting generated Chain-of-Thought CoT candidates by optimizing for winningvs. losing reasoning steps that lead to the correct answer. We train using amodified DPO loss Rafailov et al. 2023 with an additional negativelog-likelihood term which we find to be crucial. We show reasoning improvesacross repeated iterations of this scheme. While only relying on examples inthe training set our approach results in increasing accuracy forLlama-2-70B-Chat from 55.6 to 81.6 on GSM8K and 88.7 with majority votingout of 32 samples from 12.5 to 20.8 on MATH and from 77.8 to 86.7 onARC-Challenge which outperforms other Llama-2-based models not relying onadditionally sourced datasets. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.19756v1 |
|title| KAN: Kolmogorov-Arnold Networks |
|authors| Ziming LiuYixuan WangSachin VaidyaFabian RuehleJames HalversonMarin SoljačićThomas Y. HouMax Tegmark
|links| http://arxiv.org/abs/2404.19756v1 |
|updated| 2024-04-30 17:58:29 UTC |
|summary| Inspired by the Kolmogorov-Arnold representation theorem we proposeKolmogorov-Arnold Networks KANs as promising alternatives to Multi-LayerPerceptrons MLPs. While MLPs have fixed activation functions on nodesneurons KANs have learnable activation functions on edges weights.KANs have no linear weights at all -- every weight parameter is replaced by aunivariate function parametrized as a spline. We show that this seeminglysimple change makes KANs outperform MLPs in terms of accuracy andinterpretability. For accuracy much smaller KANs can achieve comparable orbetter accuracy than much larger MLPs in data fitting and PDE solving.Theoretically and empirically KANs possess faster neural scaling laws thanMLPs. For interpretability KANs can be intuitively visualized and can easilyinteract with human users. Through two examples in mathematics and physicsKANs are shown to be useful collaborators helping scientists rediscovermathematical and physical laws. In summary KANs are promising alternatives forMLPs opening opportunities for further improving todays deep learning modelswhich rely heavily on MLPs. |


| Item |Content|
| --- |---|
|idx| 2404.19753v1 |
|title| DOCCI: Descriptions of Connected and Contrasting Images |
|authors| Yasumasa OnoeSunayana RaneZachary BergerYonatan BittonJaemin ChoRoopal GargAlexander KuZarana ParekhJordi Pont-TusetGarrett TanzerSu WangJason Baldridge
|links| http://arxiv.org/abs/2404.19753v1 |
|updated| 2024-04-30 17:56:24 UTC |
|summary| Vision-language datasets are vital for both text-to-image T2I andimage-to-text I2T research. However current datasets lack descriptions withfine-grained detail that would allow for richer associations to be learned bymodels. To fill the gap we introduce Descriptions of Connected and ContrastingImages DOCCI a dataset with long human-annotated English descriptions for15k images that were taken curated and donated by a single researcher intenton capturing key challenges such as spatial relations counting textrendering world knowledge and more. We instruct human annotators to createcomprehensive descriptions for each image these average 136 words in lengthand are crafted to clearly distinguish each image from those that are relatedor similar. Each description is highly compositional and typically encompassesmultiple challenges. Through both quantitative and qualitative analyses wedemonstrate that DOCCI serves as an effective training resource forimage-to-text generation -- a PaLI 5B model finetuned on DOCCI shows equal orsuperior results compared to highly-performant larger models like LLaVA-1.5 7Band InstructBLIP 7B. Furthermore we show that DOCCI is a useful testbed fortext-to-image generation highlighting the limitations of current text-to-imagemodels in capturing long descriptions and fine details. |


| Item |Content|
| --- |---|
|idx| 2404.19749v1 |
|title| Scale-Robust Timely Asynchronous Decentralized Learning |
|authors| Purbesh MitraSennur Ulukus
|links| http://arxiv.org/abs/2404.19749v1 |
|updated| 2024-04-30 17:54:16 UTC |
|summary| We consider an asynchronous decentralized learning system which consists ofa network of connected devices trying to learn a machine learning model withoutany centralized parameter server. The users in the network have their own localtraining data which is used for learning across all the nodes in the network.The learning method consists of two processes evolving simultaneously withoutany necessary synchronization. The first process is the model update where theusers update their local model via a fixed number of stochastic gradientdescent steps. The second process is model mixing where the users communicatewith each other via randomized gossiping to exchange their models and averagethem to reach consensus. In this work we investigate the staleness criteriafor such a system which is a sufficient condition for convergence ofindividual user models. We show that for network scaling i.e. when the numberof user devices n is very large if the gossip capacity of individual usersscales as Omegalog n we can guarantee the convergence of user models infinite time. Furthermore we show that the bounded staleness can only beguaranteed by any distributed opportunistic scheme by Omegan scaling. |


| Item |Content|
| --- |---|
|idx| 2404.19739v1 |
|title| Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation |
|authors| Ian DunnDavid Ryan Koes
|links| http://arxiv.org/abs/2404.19739v1 |
|updated| 2024-04-30 17:37:21 UTC |
|summary| Deep generative models that produce novel molecular structures have thepotential to facilitate chemical discovery. Diffusion models currently achievestate of the art performance for 3D molecule generation. In this work weexplore the use of flow matching a recently proposed generative modelingframework that generalizes diffusion models for the task of de novo moleculegeneration. Flow matching provides flexibility in model design however theframework is predicated on the assumption of continuously-valued data. 3D denovo molecule generation requires jointly sampling continuous and categoricalvariables such as atom position and atom type. We extend the flow matchingframework to categorical data by constructing flows that are constrained toexist on a continuous representation of categorical data known as theprobability simplex. We call this extension SimplexFlow. We explore the use ofSimplexFlow for de novo molecule generation. However we find that inpractice a simpler approach that makes no accommodations for the categoricalnature of the data yields equivalent or superior performance. As a result ofthese experiments we present FlowMol a flow matching model for 3D de novogenerative model that achieves improved performance over prior flow matchingmethods and we raise important questions about the design of priordistributions for achieving strong performance in flow matching models. Codeand trained models for reproducing this work are available athttps://github.com/dunni3/FlowMol |


| Item |Content|
| --- |---|
|idx| 2404.19725v1 |
|title| Fairness Without Demographics in Human-Centered Federated Learning |
|authors| Roy ShailySharma HarshitSalekin Asif
|links| http://arxiv.org/abs/2404.19725v1 |
|updated| 2024-04-30 17:19:52 UTC |
|summary| Federated learning FL enables collaborative model training while preservingdata privacy making it suitable for decentralized human-centered AIapplications. However a significant research gap remains in ensuring fairnessin these systems. Current fairness strategies in FL require knowledge ofbias-creating/sensitive attributes clashing with FLs privacy principles.Moreover in human-centered datasets sensitive attributes may remain latent.To tackle these challenges we present a novel bias mitigation approachinspired by Fairness without Demographics in machine learning. The presentedapproach achieves fairness without needing knowledge of sensitive attributes byminimizing the top eigenvalue of the Hessian matrix during training ensuringequitable loss landscapes across FL participants. Notably we introduce a novelFL aggregation scheme that promotes participating models based on error ratesand loss landscape curvature attributes fostering fairness across the FLsystem. This work represents the first approach to attaining Fairness withoutDemographics in human-centered FL. Through comprehensive evaluation ourapproach demonstrates effectiveness in balancing fairness and efficacy acrossvarious real-world applications FL setups and scenarios involving single andmultiple bias-inducing factors representing a significant advancement inhuman-centered FL. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.19760v1 |
|title| Lightplane: Highly-Scalable Components for Neural 3D Fields |
|authors| Ang CaoJustin JohnsonAndrea VedaldiDavid Novotny
|links| http://arxiv.org/abs/2404.19760v1 |
|updated| 2024-04-30 17:59:51 UTC |
|summary| Contemporary 3D research particularly in reconstruction and generationheavily relies on 2D images for inputs or supervision. However current designsfor these 2D-3D mapping are memory-intensive posing a significant bottleneckfor existing methods and hindering new applications. In response we propose apair of highly scalable components for 3D neural fields: Lightplane Render andSplatter which significantly reduce memory usage in 2D-3D mapping. Theseinnovations enable the processing of vastly more and higher resolution imageswith small memory and computational costs. We demonstrate their utility invarious applications from benefiting single-scene optimization withimage-level losses to realizing a versatile pipeline for dramatically scaling3D reconstruction and generation. Code:urlhttps://github.com/facebookresearch/lightplane. |


| Item |Content|
| --- |---|
|idx| 2404.19759v1 |
|title| MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model |
|authors| Wenxun DaiLing-Hao ChenJingbo WangJinpeng LiuBo DaiYansong Tang
|links| http://arxiv.org/abs/2404.19759v1 |
|updated| 2024-04-30 17:59:47 UTC |
|summary| This work introduces MotionLCM extending controllable motion generation to areal-time level. Existing methods for spatial control in text-conditionedmotion generation suffer from significant runtime inefficiency. To address thisissue we first propose the motion latent consistency model MotionLCM formotion generation building upon the latent diffusion model MLD. By employingone-step or few-step inference we further improve the runtime efficiency ofthe motion latent diffusion model for motion generation. To ensure effectivecontrollability we incorporate a motion ControlNet within the latent space ofMotionLCM and enable explicit control signals e.g. pelvis trajectory in thevanilla motion space to control the generation process directly similar tocontrolling other latent-free diffusion models for motion generation. Byemploying these techniques our approach can generate human motions with textand control signals in real-time. Experimental results demonstrate theremarkable generation and controlling capabilities of MotionLCM whilemaintaining real-time runtime efficiency. |


| Item |Content|
| --- |---|
|idx| 2404.19758v1 |
|title| Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting |
|authors| Paul EngstlerAndrea VedaldiIro LainaChristian Rupprecht
|links| http://arxiv.org/abs/2404.19758v1 |
|updated| 2024-04-30 17:59:40 UTC |
|summary| 3D scene generation has quickly become a challenging new research directionfueled by consistent improvements of 2D generative diffusion models. Most priorwork in this area generates scenes by iteratively stitching newly generatedframes with existing geometry. These works often depend on pre-trainedmonocular depth estimators to lift the generated images into 3D fusing themwith the existing scene representation. These approaches are then oftenevaluated via a text metric measuring the similarity between the generatedimages and a given text prompt. In this work we make two fundamentalcontributions to the field of 3D scene generation. First we note that liftingimages to 3D with a monocular depth estimation model is suboptimal as itignores the geometry of the existing scene. We thus introduce a novel depthcompletion model trained via teacher distillation and self-training to learnthe 3D fusion process resulting in improved geometric coherence of the scene.Second we introduce a new benchmarking scheme for scene generation methodsthat is based on ground truth geometry and thus measures the quality of thestructure of the scene. |


| Item |Content|
| --- |---|
|idx| 2404.19753v1 |
|title| DOCCI: Descriptions of Connected and Contrasting Images |
|authors| Yasumasa OnoeSunayana RaneZachary BergerYonatan BittonJaemin ChoRoopal GargAlexander KuZarana ParekhJordi Pont-TusetGarrett TanzerSu WangJason Baldridge
|links| http://arxiv.org/abs/2404.19753v1 |
|updated| 2024-04-30 17:56:24 UTC |
|summary| Vision-language datasets are vital for both text-to-image T2I andimage-to-text I2T research. However current datasets lack descriptions withfine-grained detail that would allow for richer associations to be learned bymodels. To fill the gap we introduce Descriptions of Connected and ContrastingImages DOCCI a dataset with long human-annotated English descriptions for15k images that were taken curated and donated by a single researcher intenton capturing key challenges such as spatial relations counting textrendering world knowledge and more. We instruct human annotators to createcomprehensive descriptions for each image these average 136 words in lengthand are crafted to clearly distinguish each image from those that are relatedor similar. Each description is highly compositional and typically encompassesmultiple challenges. Through both quantitative and qualitative analyses wedemonstrate that DOCCI serves as an effective training resource forimage-to-text generation -- a PaLI 5B model finetuned on DOCCI shows equal orsuperior results compared to highly-performant larger models like LLaVA-1.5 7Band InstructBLIP 7B. Furthermore we show that DOCCI is a useful testbed fortext-to-image generation highlighting the limitations of current text-to-imagemodels in capturing long descriptions and fine details. |


| Item |Content|
| --- |---|
|idx| 2404.19752v1 |
|title| Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation |
|authors| Yunhao GeXiaohui ZengJacob Samuel HuffmanTsung-Yi LinMing-Yu LiuYin Cui
|links| http://arxiv.org/abs/2404.19752v1 |
|updated| 2024-04-30 17:55:27 UTC |
|summary| Existing automatic captioning methods for visual content face challenges suchas lack of detail content hallucination and poor instruction following. Inthis work we propose VisualFactChecker VFC a flexible training-freepipeline that generates high-fidelity and detailed captions for both 2D imagesand 3D objects. VFC consists of three steps: 1 proposal where image-to-textcaptioning models propose multiple initial captions 2 verification where alarge language model LLM utilizes tools such as object detection and VQAmodels to fact-check proposed captions 3 captioning where an LLM generatesthe final caption by summarizing caption proposals and the fact checkverification results. In this step VFC can flexibly generate captions invarious styles following complex instructions. We conduct comprehensivecaptioning evaluations using four metrics: 1 CLIP-Score for image-textsimilarity 2 CLIP-Image-Score for measuring the image-image similaritybetween the original and the reconstructed image generated by a text-to-imagemodel using the caption. 3 human study on Amazon Mechanical Turk 4 GPT-4Vfor fine-grained evaluation. Evaluation results show that VFC outperformsstate-of-the-art open-sourced captioning methods for 2D images on the COCOdataset and 3D assets on the Objaverse dataset. Our study demonstrates that bycombining open-source models into a pipeline we can attain captioningcapability comparable to proprietary models such as GPT-4V despite being over10x smaller in model size. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.19756v1 |
|title| KAN: Kolmogorov-Arnold Networks |
|authors| Ziming LiuYixuan WangSachin VaidyaFabian RuehleJames HalversonMarin SoljačićThomas Y. HouMax Tegmark
|links| http://arxiv.org/abs/2404.19756v1 |
|updated| 2024-04-30 17:58:29 UTC |
|summary| Inspired by the Kolmogorov-Arnold representation theorem we proposeKolmogorov-Arnold Networks KANs as promising alternatives to Multi-LayerPerceptrons MLPs. While MLPs have fixed activation functions on nodesneurons KANs have learnable activation functions on edges weights.KANs have no linear weights at all -- every weight parameter is replaced by aunivariate function parametrized as a spline. We show that this seeminglysimple change makes KANs outperform MLPs in terms of accuracy andinterpretability. For accuracy much smaller KANs can achieve comparable orbetter accuracy than much larger MLPs in data fitting and PDE solving.Theoretically and empirically KANs possess faster neural scaling laws thanMLPs. For interpretability KANs can be intuitively visualized and can easilyinteract with human users. Through two examples in mathematics and physicsKANs are shown to be useful collaborators helping scientists rediscovermathematical and physical laws. In summary KANs are promising alternatives forMLPs opening opportunities for further improving todays deep learning modelswhich rely heavily on MLPs. |


| Item |Content|
| --- |---|
|idx| 2404.19719v1 |
|title| The lazy (NTK) and rich ($μ$P) regimes: a gentle tutorial |
|authors| Dhruva Karkada
|links| http://arxiv.org/abs/2404.19719v1 |
|updated| 2024-04-30 17:11:12 UTC |
|summary| A central theme of the modern machine learning paradigm is that larger neuralnetworks achieve better performance on a variety of metrics. Theoreticalanalyses of these overparameterized models have recently centered aroundstudying very wide neural networks. In this tutorial we provide a nonrigorousbut illustrative derivation of the following fact: in order to train widenetworks effectively there is only one degree of freedom in choosinghyperparameters such as the learning rate and the size of the initial weights.This degree of freedom controls the richness of training behavior: at minimumthe wide network trains lazily like a kernel machine and at maximum itexhibits feature learning in the so-called muP regime. In this paper weexplain this richness scale synthesize recent research results into a coherentwhole offer new perspectives and intuitions and provide empirical evidencesupporting our claims. In doing so we hope to encourage further study of therichness scale as it may be key to developing a scientific theory of featurelearning in practical deep neural networks. |


| Item |Content|
| --- |---|
|idx| 2404.19661v1 |
|title| PCA for Point Processes |
|authors| Franck PicardVincent RivoirardAngelina RocheVictor Panaretos
|links| http://arxiv.org/abs/2404.19661v1 |
|updated| 2024-04-30 15:57:18 UTC |
|summary| We introduce a novel statistical framework for the analysis of replicatedpoint processes that allows for the study of point pattern variability at apopulation level. By treating point process realizations as random measures weadopt a functional analysis perspective and propose a form of functionalPrincipal Component Analysis fPCA for point processes. The originality of ourmethod is to base our analysis on the cumulative mass functions of the randommeasures which gives us a direct and interpretable analysis. Key theoreticalcontributions include establishing a Karhunen-Loeve expansion for therandom measures and a Mercer Theorem for covariance measures. We establishconvergence in a strong sense and introduce the concept of principal measureswhich can be seen as latent processes governing the dynamics of the observedpoint patterns. We propose an easy-to-implement estimation strategy ofeigenelements for which parametric rates are achieved. We fully characterizethe solutions of our approach to Poisson and Hawkes processes and validate ourmethodology via simulations and diverse applications in seismology single-cellbiology and neurosiences demonstrating its versatility and effectiveness. Ourmethod is implemented in the pppca R-package. |


| Item |Content|
| --- |---|
|idx| 2404.19620v1 |
|title| Be Aware of the Neighborhood Effect: Modeling Selection Bias under Interference |
|authors| Haoxuan LiChunyuan ZhengSihao DingPeng WuZhi GengFuli FengXiangnan He
|links| http://arxiv.org/abs/2404.19620v1 |
|updated| 2024-04-30 15:20:41 UTC |
|summary| Selection bias in recommender system arises from the recommendation processof system filtering and the interactive process of user selection. Manyprevious studies have focused on addressing selection bias to achieve unbiasedlearning of the prediction model but ignore the fact that potential outcomesfor a given user-item pair may vary with the treatments assigned to otheruser-item pairs named neighborhood effect. To fill the gap this paperformally formulates the neighborhood effect as an interference problem from theperspective of causal inference and introduces a treatment representation tocapture the neighborhood effect. On this basis we propose a novel ideal lossthat can be used to deal with selection bias in the presence of neighborhoodeffect. We further develop two new estimators for estimating the proposed idealloss. We theoretically establish the connection between the proposed andprevious debiasing methods ignoring the neighborhood effect showing that theproposed methods can achieve unbiased learning when both selection bias andneighborhood effect are present while the existing methods are biased.Extensive semi-synthetic and real-world experiments are conducted todemonstrate the effectiveness of the proposed methods. |


| Item |Content|
| --- |---|
|idx| 2404.19557v1 |
|title| Neural Dynamic Data Valuation |
|authors| Zhangyong LiangHuanhuan GaoJi Zhang
|links| http://arxiv.org/abs/2404.19557v1 |
|updated| 2024-04-30 13:39:26 UTC |
|summary| Data constitute the foundational component of the data economy and itsmarketplaces. Efficient and fair data valuation has emerged as a topic ofsignificant interest. Many approaches based on marginal contribution haveshown promising results in various downstream tasks. However they are wellknown to be computationally expensive as they require training a large numberof utility functions which are used to evaluate the usefulness or value of agiven dataset for a specific purpose. As a result it has been recognized asinfeasible to apply these methods to a data marketplace involving large-scaledatasets. Consequently a critical issue arises: how can the re-training of theutility function be avoided To address this issue we propose a novel datavaluation method from the perspective of optimal control named the neuraldynamic data valuation NDDV. Our method has solid theoretical interpretationsto accurately identify the data valuation via the sensitivity of the dataoptimal control state. In addition we implement a data re-weighting strategyto capture the unique features of data points ensuring fairness through theinteraction between data points and the mean-field states. Notably our methodrequires only training once to estimate the value of all data pointssignificantly improving the computational efficiency. We conduct comprehensiveexperiments using different datasets and tasks. The results demonstrate thatthe proposed NDDV method outperforms the existing state-of-the-art datavaluation methods in accurately identifying data points with either high or lowvalues and is more computationally efficient. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.19738v1 |
|title| DiaryHelper: Exploring the Use of an Automatic Contextual Information Recording Agent for Elicitation Diary Study |
|authors| Junze LiChangyang HeJiaxiong HuBoyang JiaAlon HalevyXiaojuan Ma
|links| http://arxiv.org/abs/2404.19738v1 |
|updated| 2024-04-30 17:36:06 UTC |
|summary| Elicitation diary studies a type of qualitative longitudinal researchmethod involve participants to self-report aspects of events of interest attheir occurrences as memory cues for providing details and insights duringpost-study interviews. However due to time constraints and lack of motivationparticipants diary entries may be vague or incomplete impairing their laterrecall. To address this challenge we designed an automatic contextualinformation recording agent DiaryHelper based on the theory of episodicmemory. DiaryHelper can predict five dimensions of contextual information andconfirm with participants. We evaluated the use of DiaryHelper in both therecording period and the elicitation interview through a within-subject studyN12 over a period of two weeks. Our results demonstrated that DiaryHelpercan assist participants in capturing abundant and accurate contextualinformation without significant burden leading to a more detailed recall ofrecorded events and providing greater insights. |


| Item |Content|
| --- |---|
|idx| 2404.19729v1 |
|title| A Framework for Leveraging Human Computation Gaming to Enhance Knowledge Graphs for Accuracy Critical Generative AI Applications |
|authors| Steph BuongiornoCorey Clark
|links| http://arxiv.org/abs/2404.19729v1 |
|updated| 2024-04-30 17:24:55 UTC |
|summary| External knowledge graphs KGs can be used to augment large language modelsLLMs while simultaneously providing an explainable knowledge base of factsthat can be inspected by a human. This approach may be particularly valuable indomains where explainability is critical like human trafficking data analysis.However creating KGs can pose challenges. KGs parsed from documents maycomprise explicit connections those directly stated by a document but missimplicit connections those obvious to a human although not directly stated.To address these challenges this preliminary research introduces the GAME-KGframework standing for Gaming for Augmenting Metadata and Enhancing KnowledgeGraphs. GAME-KG is a federated approach to modifying explicit as well asimplicit connections in KGs by using crowdsourced feedback collected throughvideo games. GAME-KG is shown through two demonstrations: a Unity test scenariofrom Dark Shadows a video game that collects feedback on KGs parsed from USDepartment of Justice DOJ Press Releases on human trafficking and afollowing experiment where OpenAIs GPT-4 is prompted to answer questions basedon a modified and unmodified KG. Initial results suggest that GAME-KG can be aneffective framework for enhancing KGs while simultaneously providing anexplainable set of structured facts verified by humans. |


| Item |Content|
| --- |---|
|idx| 2404.19708v1 |
|title| Harmonic LLMs are Trustworthy |
|authors| Nicholas S. KerstingMohammad RahmanSuchismitha VedalaYang Wang
|links| http://arxiv.org/abs/2404.19708v1 |
|updated| 2024-04-30 17:00:32 UTC |
|summary| We introduce an intuitive method to test the robustness stability andexplainability of any black-box LLM in real-time based upon the localdeviation from harmoniticity denoted as gamma. To the best of our knowledgethis is the first completely model-agnostic and unsupervised method ofmeasuring the robustness of any given response from an LLM based upon themodel itself conforming to a purely mathematical standard. We conduct humanannotation experiments to show the positive correlation of gamma with falseor misleading answers and demonstrate that following the gradient of gammain stochastic gradient ascent efficiently exposes adversarial prompts.Measuring gamma across thousands of queries in popular LLMs GPT-4 ChatGPTClaude-2.1 Mixtral-8x7B Smaug-72B Llama2-7B and MPT-7B allows us toestimate the liklihood of wrong or hallucinatory answers automatically andquantitatively rank the reliability of these models in various objectivedomains Web QA TruthfulQA and Programming QA. Across all models and domainstested human ratings confirm that gamma to 0 indicates trustworthinessand the low-gamma leaders among these models are GPT-4 ChatGPT andSmaug-72B. |


| Item |Content|
| --- |---|
|idx| 2404.19693v1 |
|title| SwipeGANSpace: Swipe-to-Compare Image Generation via Efficient Latent Space Exploration |
|authors| Yuto NakashimaMingzhe YangYukino Baba
|links| http://dx.doi.org/10.1145/3640543.3645141 |
|updated| 2024-04-30 16:37:27 UTC |
|summary| Generating preferred images using generative adversarial networks GANs ischallenging owing to the high-dimensional nature of latent space. In thisstudy we propose a novel approach that uses simple user-swipe interactions togenerate preferred images for users. To effectively explore the latent spacewith only swipe interactions we apply principal component analysis to thelatent space of the StyleGAN creating meaningful subspaces. We use amulti-armed bandit algorithm to decide the dimensions to explore focusing onthe preferences of the user. Experiments show that our method is more efficientin generating preferred images than the baseline methods. Furthermore changesin preferred images during image generation or the display of entirelydifferent image styles were observed to provide new inspirations subsequentlyaltering user preferences. This highlights the dynamic nature of userpreferences which our proposed approach recognizes and enhances. |


| Item |Content|
| --- |---|
|idx| 2404.19629v1 |
|title| The Drawback of Insight: Detailed Explanations Can Reduce Agreement with XAI |
|authors| Sabid Bin Habib PiasAlicia FreelTimothy TrammelTaslima AkterDonald WilliamsonApu Kapadia
|links| http://arxiv.org/abs/2404.19629v1 |
|updated| 2024-04-30 15:29:01 UTC |
|summary| With the emergence of Artificial Intelligence AI-based decision-makingexplanations help increase new technology adoption through enhanced trust andreliability. However our experimental study challenges the notion that everyuser universally values explanations. We argue that the agreement with AIsuggestions whether accompanied by explanations or not is influenced byindividual differences in personality traits and the users comfort withtechnology. We found that people with higher neuroticism and lowertechnological comfort showed more agreement with the recommendations withoutexplanations. As more users become exposed to eXplainable AI XAI and AI-basedsystems we argue that the XAI design should not provide explanations for userswith high neuroticism and low technology comfort. Prioritizing userpersonalities in XAI systems will help users become better collaborators of AIsystems. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.19749v1 |
|title| Scale-Robust Timely Asynchronous Decentralized Learning |
|authors| Purbesh MitraSennur Ulukus
|links| http://arxiv.org/abs/2404.19749v1 |
|updated| 2024-04-30 17:54:16 UTC |
|summary| We consider an asynchronous decentralized learning system which consists ofa network of connected devices trying to learn a machine learning model withoutany centralized parameter server. The users in the network have their own localtraining data which is used for learning across all the nodes in the network.The learning method consists of two processes evolving simultaneously withoutany necessary synchronization. The first process is the model update where theusers update their local model via a fixed number of stochastic gradientdescent steps. The second process is model mixing where the users communicatewith each other via randomized gossiping to exchange their models and averagethem to reach consensus. In this work we investigate the staleness criteriafor such a system which is a sufficient condition for convergence ofindividual user models. We show that for network scaling i.e. when the numberof user devices n is very large if the gossip capacity of individual usersscales as Omegalog n we can guarantee the convergence of user models infinite time. Furthermore we show that the bounded staleness can only beguaranteed by any distributed opportunistic scheme by Omegan scaling. |


| Item |Content|
| --- |---|
|idx| 2404.19745v1 |
|title| Analyzing Transport Policies in Developing Countries with ABM |
|authors| Kathleen Salazar-SernaLorena CadavidCarlos Franco
|links| http://arxiv.org/abs/2404.19745v1 |
|updated| 2024-04-30 17:47:30 UTC |
|summary| Deciphering travel behavior and mode choices is a critical aspect ofeffective urban transportation system management particularly in developingcountries where unique socio-economic and cultural conditions complicatedecision-making. Agent-based simulations offer a valuable tool for modelingtransportation systems enabling a nuanced understanding and policy impactevaluation. This work aims to shed light on the effects of transport policiesand analyzes travel behavior by simulating agents making mode choices for theirdaily commutes. Agents gather information from the environment and their socialnetwork to assess the optimal transport option based on personal satisfactioncriteria. Our findings stemming from simulating a free-fare policy for publictransit in a developing-country city reveal a significant influence ondecision-making fostering public service use while positively influencingpollution levels accident rates and travel speed. |


| Item |Content|
| --- |---|
|idx| 2404.19564v1 |
|title| Time, Travel, and Energy in the Uniform Dispersion Problem |
|authors| Michael AmirAlfred M. Bruckstein
|links| http://arxiv.org/abs/2404.19564v1 |
|updated| 2024-04-30 13:51:44 UTC |
|summary| We investigate the algorithmic problem of uniformly dispersing a swarm ofrobots in an unknown gridlike environment. In this setting our goal is tocomprehensively study the relationships between performance metrics and robotcapabilities. We introduce a formal model comparing dispersion algorithms basedon makespan traveled distance energy consumption sensing communication andmemory. Using this framework we classify several uniform dispersion algorithmsaccording to their capability requirements and performance. We prove that whilemakespan and travel can be minimized in all environments energy cannot aslong as the swarms sensing range is bounded. In contrast we show that energycan be minimized even by simple ant-like robots in synchronous settings andasymptotically minimized in asynchronous settings provided the environment istopologically simply connected. Our findings offer insights into fundamentallimitations that arise when designing swarm robotics systems for exploringunknown environments highlighting the impact of environments topology on thefeasibility of energy-efficient dispersion. |


| Item |Content|
| --- |---|
|idx| 2404.19547v1 |
|title| Distributed Traffic Signal Control via Coordinated Maximum Pressure-plus-Penalty |
|authors| Vinzenz TütschZhiyu HeFlorian DörflerKenan Zhang
|links| http://arxiv.org/abs/2404.19547v1 |
|updated| 2024-04-30 13:16:05 UTC |
|summary| This paper develops an adaptive traffic control policy inspired by MaximumPressure MP while imposing coordination across intersections. The proposedCoordinated Maximum Pressure-plus-Penalty CMPP control policy features alocal objective for each intersection that consists of the total pressurewithin the neighborhood and a penalty accounting for the queue capacities andcontinuous green time for certain movements. The corresponding control task isreformulated as a distributed optimization problem and solved via twocustomized algorithms: one based on the alternating direction method ofmultipliers ADMM and the other follows a greedy heuristic augmented with amajority vote. CMPP not only provides a theoretical guarantee of queuingnetwork stability but also outperforms several benchmark controllers insimulations on a large-scale real traffic network with lower average travel andwaiting time per vehicle as well as less network congestion. Furthermore CPMMwith the greedy algorithm enjoys comparable computational efficiency as fullydecentralized controllers without significantly compromising the controlperformance which highlights its great potential for real-world deployment. |


| Item |Content|
| --- |---|
|idx| 2404.19518v1 |
|title| MGCBS: An Optimal and Efficient Algorithm for Solving Multi-Goal Multi-Agent Path Finding Problem |
|authors| Mingkai TangYuanhang LiHongji LiuYingbing ChenMing LiuLujia Wang
|links| http://arxiv.org/abs/2404.19518v1 |
|updated| 2024-04-30 12:49:54 UTC |
|summary| With the expansion of the scale of robotics applications the multi-goalmulti-agent pathfinding MG-MAPF problem began to gain widespread attention.This problem requires each agent to visit pre-assigned multiple goal points atleast once without conflict. Some previous methods have been proposed to solvethe MG-MAPF problem based on Decoupling the goal Vertex visiting order searchand the Single-agent pathfinding DVS. However this paper demonstrates thatthe methods based on DVS cannot always obtain the optimal solution. To obtainthe optimal result we propose the Multi-Goal Conflict-Based Search MGCBSwhich is based on Decoupling the goal Safe interval visiting order search andthe Single-agent pathfinding DSS. Additionally we present theTime-Interval-Space Forest TIS Forest to enhance the efficiency of MGCBS bymaintaining the shortest paths from any start point at any start time step toeach safe interval at the goal points. The experiment demonstrates that ourmethod can consistently obtain optimal results and execute up to 7 times fasterthan the state-of-the-art method in our evaluation. |


