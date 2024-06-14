# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.09411v1 |
|title| MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding |
|authors| Fei WangXingyu FuJames Y. HuangZekun LiQin LiuXiaogeng LiuMingyu Derek MaNan XuWenxuan ZhouKai ZhangTianyi Lorena YanWenjie Jacky MoHsiang-Hui LiuPan LuChunyuan LiChaowei XiaoKai-Wei ChangDan RothSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.09411v1 |
|updated| 2024-06-13 17:59:52 UTC |
|summary| We introduce MuirBench a comprehensive benchmark that focuses on robustmulti-image understanding capabilities of multimodal LLMs. MuirBench consistsof 12 diverse multi-image tasks e.g. scene understanding ordering thatinvolve 10 categories of multi-image relations e.g. multiview temporalrelations. Comprising 11264 images and 2600 multiple-choice questionsMuirBench is created in a pairwise manner where each standard instance ispaired with an unanswerable variant that has minimal semantic differences inorder for a reliable assessment. Evaluated upon 20 recent multi-modal LLMs ourresults reveal that even the best-performing models like GPT-4o and Gemini Profind it challenging to solve MuirBench achieving 68.0 and 49.3 in accuracy.Open-source multimodal LLMs trained on single images can hardly generalize tomulti-image questions hovering below 33.3 in accuracy. These resultshighlight the importance of MuirBench in encouraging the community to developmultimodal LLMs that can look beyond a single image suggesting potentialpathways for future improvements. |


| Item |Content|
| --- |---|
|idx| 2406.09403v1 |
|title| Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models |
|authors| Yushi HuWeijia ShiXingyu FuDan RothMari OstendorfLuke ZettlemoyerNoah A SmithRanjay Krishna
|links| http://arxiv.org/abs/2406.09403v1 |
|updated| 2024-06-13 17:59:31 UTC |
|summary| Humans draw to facilitate reasoning: we draw auxiliary lines when solvinggeometry problems we mark and circle when reasoning on maps we use sketchesto amplify our ideas and relieve our limited-capacity working memory. Howeversuch actions are missing in current multimodal language models LMs. Currentchain-of-thought and tool-use paradigms only use text as intermediate reasoningsteps. In this work we introduce Sketchpad a framework that gives multimodalLMs a visual sketchpad and tools to draw on the sketchpad. The LM conductsplanning and reasoning according to the visual artifacts it has drawn.Different from prior work which uses text-to-image models to enable LMs todraw Sketchpad enables LMs to draw with lines boxes marks etc. which iscloser to human sketching and better facilitates reasoning. Sketchpad can alsouse specialist vision models during the sketching process e.g. draw boundingboxes with object detection models draw masks with segmentation models tofurther enhance visual perception and reasoning. We experiment with a widerange of math tasks including geometry functions graphs and chess andcomplex visual reasoning tasks. Sketchpad substantially improves performance onall tasks over strong base models with no sketching yielding an average gainof 12.7 on math tasks and 8.6 on vision tasks. GPT-4o with Sketchpad sets anew state of the art on all tasks including VBench 80.3 BLINK spatialreasoning 83.9 and visual correspondence 80.8. All codes and data are inhttps://visualsketchpad.github.io/. |


| Item |Content|
| --- |---|
|idx| 2406.09393v1 |
|title| Improving Autoregressive Training with Dynamic Oracles |
|authors| Jianing YangHarshine VisvanathanYilin WangXinyi HuMatthew Gormley
|links| http://arxiv.org/abs/2406.09393v1 |
|updated| 2024-06-13 17:59:09 UTC |
|summary| Many tasks within NLP can be framed as sequential decision problems rangingfrom sequence tagging to text generation. However for many tasks the standardtraining methods including maximum likelihood teacher forcing and scheduledsampling suffer from exposure bias and a mismatch between metrics employedduring training and inference. DAgger provides a solution to mitigate theseproblems yet it requires a metric-specific dynamic oracle algorithm whichdoes not exist for many common metrics like span-based F1 ROUGE and BLEU. Inthis paper we develop these novel dynamic oracles and show they maintainDAggers no-regret guarantee for decomposable metrics like span-based F1. Weevaluate the algorithms performance on named entity recognition NER textsummarization and machine translation MT. While DAgger with dynamic oracleyields less favorable results in our MT experiments it outperforms thebaseline techniques in NER and text summarization. |


| Item |Content|
| --- |---|
|idx| 2406.09345v1 |
|title| DiscreteSLU: A Large Language Model with Self-Supervised Discrete Speech Units for Spoken Language Understanding |
|authors| Suwon ShonKwangyoun KimYi-Te HsuPrashant SridharShinji WatanabeKaren Livescu
|links| http://arxiv.org/abs/2406.09345v1 |
|updated| 2024-06-13 17:28:13 UTC |
|summary| The integration of pre-trained text-based large language models LLM withspeech input has enabled instruction-following capabilities for diverse speechtasks. This integration requires the use of a speech encoder a speech adapterand an LLM trained on diverse tasks. We propose the use of discrete speechunits DSU rather than continuous-valued speech encoder outputs that areconverted to the LLM token embedding space using the speech adapter. Wegenerate DSU using a self-supervised speech encoder followed by k-meansclustering. The proposed model shows robust performance on speech inputs fromseen/unseen domains and instruction-following capability in spoken questionanswering. We also explore various types of DSU extracted from different layersof the self-supervised speech encoder as well as Mel frequency CepstralCoefficients MFCC. Our findings suggest that the ASR task and datasets arenot crucial in instruction-tuning for spoken question answering tasks. |


| Item |Content|
| --- |---|
|idx| 2406.09334v1 |
|title| ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models |
|authors| David AnugrahaGenta Indra WinataChenyue LiPatrick Amadeus IrawanEn-Shiun Annie Lee
|links| http://arxiv.org/abs/2406.09334v1 |
|updated| 2024-06-13 17:15:33 UTC |
|summary| Performance prediction is a method to estimate the performance ofmultilingual language models LMs mitigating computational costs associatedwith model capacity and data for fine-tuning. Our paper introduces ProxyLM ascalable framework for predicting LM performance using proxy models inmultilingual tasks. These proxy models act as surrogates approximating theperformance of fine-tuned LMs on specific downstream natural languageprocessing NLP tasks. By leveraging proxy models ProxyLM significantlyreduces computational overhead on task evaluations achieving up to a 37.08xspeedup compared to traditional methods even with our smallest proxy models.Additionally our methodology showcases adaptability to previously unseenlanguages in pre-trained LMs outperforming the state-of-the-art performance by1.89x as measured by root-mean-square-error RMSE. This framework streamlinesmodel selection enabling efficient deployment and iterative LM enhancementswithout extensive computational resources. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.09412v1 |
|title| Explore the Limits of Omni-modal Pretraining at Scale |
|authors| Yiyuan ZhangHandong LiJing LiuXiangyu Yue
|links| http://arxiv.org/abs/2406.09412v1 |
|updated| 2024-06-13 17:59:53 UTC |
|summary| We propose to build omni-modal intelligence which is capable ofunderstanding any modality and learning universal representations. In specificwe propose a scalable pretraining paradigm named Multimodal Context MiCowhich can scale up the numbers of modalities and amount of data together withthe model parameters in the pretraining process. With MiCo the pretrainedmodels show significant emergent abilities in multimodal learning which areevaluated on the following tasks: i single-modality perception benchmarks of10 different modalities ii 25 cross-modality understanding tasks ofretrieval question-answering captioning and iii 18 multimodal largelanguage model benchmarks. Our models establish 37 new records forstate-of-the-art performance. We hope that our research could contribute to thedevelopment of omni-modal intelligence. Code and Models are athttps://github.com/invictus717/MiCo |


| Item |Content|
| --- |---|
|idx| 2406.09411v1 |
|title| MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding |
|authors| Fei WangXingyu FuJames Y. HuangZekun LiQin LiuXiaogeng LiuMingyu Derek MaNan XuWenxuan ZhouKai ZhangTianyi Lorena YanWenjie Jacky MoHsiang-Hui LiuPan LuChunyuan LiChaowei XiaoKai-Wei ChangDan RothSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.09411v1 |
|updated| 2024-06-13 17:59:52 UTC |
|summary| We introduce MuirBench a comprehensive benchmark that focuses on robustmulti-image understanding capabilities of multimodal LLMs. MuirBench consistsof 12 diverse multi-image tasks e.g. scene understanding ordering thatinvolve 10 categories of multi-image relations e.g. multiview temporalrelations. Comprising 11264 images and 2600 multiple-choice questionsMuirBench is created in a pairwise manner where each standard instance ispaired with an unanswerable variant that has minimal semantic differences inorder for a reliable assessment. Evaluated upon 20 recent multi-modal LLMs ourresults reveal that even the best-performing models like GPT-4o and Gemini Profind it challenging to solve MuirBench achieving 68.0 and 49.3 in accuracy.Open-source multimodal LLMs trained on single images can hardly generalize tomulti-image questions hovering below 33.3 in accuracy. These resultshighlight the importance of MuirBench in encouraging the community to developmultimodal LLMs that can look beyond a single image suggesting potentialpathways for future improvements. |


| Item |Content|
| --- |---|
|idx| 2406.09410v1 |
|title| Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach |
|authors| Yansheng LiLinlin WangTingzhu WangXue YangJunwei LuoQi WangYouming DengWenbin WangXian SunHaifeng LiBo DangYongjun ZhangYi YuJunchi Yan
|links| http://arxiv.org/abs/2406.09410v1 |
|updated| 2024-06-13 17:59:51 UTC |
|summary| Scene graph generation SGG in satellite imagery SAI benefits promotingintelligent understanding of geospatial scenarios from perception to cognition.In SAI objects exhibit great variations in scales and aspect ratios and thereexist rich relationships between objects even between spatially disjointobjects which makes it necessary to holistically conduct SGG in large-sizevery-high-resolution VHR SAI. However the lack of SGG datasets withlarge-size VHR SAI has constrained the advancement of SGG in SAI. Due to thecomplexity of large-size VHR SAI mining triplets subject relationshipobject in large-size VHR SAI heavily relies on long-range contextualreasoning. Consequently SGG models designed for small-size natural imagery arenot directly applicable to large-size VHR SAI. To address the scarcity ofdatasets this paper constructs a large-scale dataset for SGG in large-size VHRSAI with image sizes ranging from 512 x 768 to 27860 x 31096 pixels namedRSG encompassing over 210000 objects and more than 400000 triplets. Torealize SGG in large-size VHR SAI we propose a context-aware cascade cognitionCAC framework to understand SAI at three levels: object detection OBD pairpruning and relationship prediction. As a fundamental prerequisite for SGG inlarge-size SAI a holistic multi-class object detection network HOD-Net thatcan flexibly integrate multi-scale contexts is proposed. With the considerationthat there exist a huge amount of object pairs in large-size SAI but only aminority of object pairs contain meaningful relationships we design a pairproposal generation PPG network via adversarial reconstruction to selecthigh-value pairs. Furthermore a relationship prediction network withcontext-aware messaging RPCM is proposed to predict the relationship types ofthese pairs. |


| Item |Content|
| --- |---|
|idx| 2406.09406v1 |
|title| 4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities |
|authors| Roman BachmannOğuzhan Fatih KarDavid MizrahiAli GarjaniMingfei GaoDavid GriffithsJiaming HuAfshin DehghanAmir Zamir
|links| http://arxiv.org/abs/2406.09406v1 |
|updated| 2024-06-13 17:59:42 UTC |
|summary| Current multimodal and multitask foundation models like 4M or UnifiedIO showpromising results but in practice their out-of-the-box abilities to acceptdiverse inputs and perform diverse tasks are limited by the usually rathersmall number of modalities and tasks they are trained on. In this paper weexpand upon the capabilities of them by training a single model on tens ofhighly diverse modalities and by performing co-training on large-scalemultimodal datasets and text corpora. This includes training on severalsemantic and geometric modalities feature maps from recent state of the artmodels like DINOv2 and ImageBind pseudo labels of specialist models like SAMand 4DHumans and a range of new modalities that allow for novel ways tointeract with the model and steer the generation for example image metadata orcolor palettes. A crucial step in this process is performing discretetokenization on various modalities whether they are image-like neural networkfeature maps vectors structured data like instance segmentation or humanposes or data that can be represented as text. Through this we expand on theout-of-the-box capabilities of multimodal models and specifically show thepossibility of training one model to solve at least 3x more tasks/modalitiesthan existing ones and doing so without a loss in performance. This enablesmore fine-grained and controllable multimodal generation capabilities andallows us to study the distillation of models trained on diverse data andobjectives into a unified model. We successfully scale the training to a threebillion parameter model using tens of modalities and different datasets. Theresulting models and training code are open sourced at 4m.epfl.ch. |


| Item |Content|
| --- |---|
|idx| 2406.09404v1 |
|title| ConsistDreamer: 3D-Consistent 2D Diffusion for High-Fidelity Scene Editing |
|authors| Jun-Kun ChenSamuel Rota BulòNorman MüllerLorenzo PorziPeter KontschiederYu-Xiong Wang
|links| http://arxiv.org/abs/2406.09404v1 |
|updated| 2024-06-13 17:59:32 UTC |
|summary| This paper proposes ConsistDreamer - a novel framework that lifts 2Ddiffusion models with 3D awareness and 3D consistency thus enablinghigh-fidelity instruction-guided scene editing. To overcome the fundamentallimitation of missing 3D consistency in 2D diffusion models our key insight isto introduce three synergetic strategies that augment the input of the 2Ddiffusion model to become 3D-aware and to explicitly enforce 3D consistencyduring the training process. Specifically we design surrounding views ascontext-rich input for the 2D diffusion model and generate 3D-consistentstructured noise instead of image-independent noise. Moreover we introduceself-supervised consistency-enforcing training within the per-scene editingprocedure. Extensive evaluation shows that our ConsistDreamer achievesstate-of-the-art performance for instruction-guided scene editing acrossvarious scenes and editing instructions particularly in complicatedlarge-scale indoor scenes from ScanNet with significantly improved sharpnessand fine-grained textures. Notably ConsistDreamer stands as the first workcapable of successfully editing complex e.g. plaid/checkered patterns. Ourproject page is at immortalco.github.io/ConsistDreamer. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.09415v1 |
|title| An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels |
|authors| Duy-Kien NguyenMahmoud AssranUnnat JainMartin R. OswaldCees G. M. SnoekXinlei Chen
|links| http://arxiv.org/abs/2406.09415v1 |
|updated| 2024-06-13 17:59:58 UTC |
|summary| This work does not introduce a new method. Instead we present an interestingfinding that questions the necessity of the inductive bias -- locality inmodern computer vision architectures. Concretely we find that vanillaTransformers can operate by directly treating each individual pixel as a tokenand achieve highly performant results. This is substantially different from thepopular design in Vision Transformer which maintains the inductive bias fromConvNets towards local neighborhoods e.g. by treating each 16x16 patch as atoken. We mainly showcase the effectiveness of pixels-as-tokens across threewell-studied tasks in computer vision: supervised learning for objectclassification self-supervised learning via masked autoencoding and imagegeneration with diffusion models. Although directly operating on individualpixels is less computationally practical we believe the community must beaware of this surprising piece of knowledge when devising the next generationof neural architectures for computer vision. |


| Item |Content|
| --- |---|
|idx| 2406.09417v1 |
|title| Rethinking Score Distillation as a Bridge Between Image Distributions |
|authors| David McAllisterSongwei GeJia-Bin HuangDavid W. JacobsAlexei A. EfrosAleksander HolynskiAngjoo Kanazawa
|links| http://arxiv.org/abs/2406.09417v1 |
|updated| 2024-06-13 17:59:58 UTC |
|summary| Score distillation sampling SDS has proven to be an important toolenabling the use of large-scale diffusion priors for tasks operating indata-poor domains. Unfortunately SDS has a number of characteristic artifactsthat limit its usefulness in general-purpose applications. In this paper wemake progress toward understanding the behavior of SDS and its variants byviewing them as solving an optimal-cost transport path from a sourcedistribution to a target distribution. Under this new interpretation thesemethods seek to transport corrupted images source to the natural imagedistribution target. We argue that current methods characteristic artifactsare caused by 1 linear approximation of the optimal path and 2 poorestimates of the source distribution. We show that calibrating the textconditioning of the source distribution can produce high-quality generation andtranslation results with little extra overhead. Our method can be easilyapplied across many domains matching or beating the performance of specializedmethods. We demonstrate its utility in text-to-2D text-based NeRFoptimization translating paintings to real images optical illusiongeneration and 3D sketch-to-real. We compare our method to existing approachesfor score distillation sampling and show that it can produce high-frequencydetails with realistic colors. |


| Item |Content|
| --- |---|
|idx| 2406.09413v1 |
|title| Interpreting the Weight Space of Customized Diffusion Models |
|authors| Amil DravidYossi GandelsmanKuan-Chieh WangRameen AbdalGordon WetzsteinAlexei A. EfrosKfir Aberman
|links| http://arxiv.org/abs/2406.09413v1 |
|updated| 2024-06-13 17:59:56 UTC |
|summary| We investigate the space of weights spanned by a large collection ofcustomized diffusion models. We populate this space by creating a dataset ofover 60000 models each of which is a base model fine-tuned to insert adifferent persons visual identity. We model the underlying manifold of theseweights as a subspace which we term weights2weights. We demonstrate threeimmediate applications of this space -- sampling editing and inversion.First as each point in the space corresponds to an identity sampling a set ofweights from it results in a model encoding a novel identity. Next we findlinear directions in this space corresponding to semantic edits of the identitye.g. adding a beard. These edits persist in appearance across generatedsamples. Finally we show that inverting a single image into this spacereconstructs a realistic identity even if the input image is out ofdistribution e.g. a painting. Our results indicate that the weight space offine-tuned diffusion models behaves as an interpretable latent space ofidentities. |


| Item |Content|
| --- |---|
|idx| 2406.09412v1 |
|title| Explore the Limits of Omni-modal Pretraining at Scale |
|authors| Yiyuan ZhangHandong LiJing LiuXiangyu Yue
|links| http://arxiv.org/abs/2406.09412v1 |
|updated| 2024-06-13 17:59:53 UTC |
|summary| We propose to build omni-modal intelligence which is capable ofunderstanding any modality and learning universal representations. In specificwe propose a scalable pretraining paradigm named Multimodal Context MiCowhich can scale up the numbers of modalities and amount of data together withthe model parameters in the pretraining process. With MiCo the pretrainedmodels show significant emergent abilities in multimodal learning which areevaluated on the following tasks: i single-modality perception benchmarks of10 different modalities ii 25 cross-modality understanding tasks ofretrieval question-answering captioning and iii 18 multimodal largelanguage model benchmarks. Our models establish 37 new records forstate-of-the-art performance. We hope that our research could contribute to thedevelopment of omni-modal intelligence. Code and Models are athttps://github.com/invictus717/MiCo |


| Item |Content|
| --- |---|
|idx| 2406.09408v1 |
|title| Data Attribution for Text-to-Image Models by Unlearning Synthesized Images |
|authors| Sheng-Yu WangAaron HertzmannAlexei A. EfrosJun-Yan ZhuRichard Zhang
|links| http://arxiv.org/abs/2406.09408v1 |
|updated| 2024-06-13 17:59:44 UTC |
|summary| The goal of data attribution for text-to-image models is to identify thetraining images that most influence the generation of a new image. We candefine influence by saying that for a given output if a model is retrainedfrom scratch without that outputs most influential images the model shouldthen fail to generate that output image. Unfortunately directly searching forthese influential images is computationally infeasible since it would requirerepeatedly retraining from scratch. We propose a new approach that efficientlyidentifies highly-influential images. Specifically we simulate unlearning thesynthesized image proposing a method to increase the training loss on theoutput image without catastrophic forgetting of other unrelated concepts.Then we find training images that are forgotten by proxy identifying oneswith significant loss deviations after the unlearning process and label theseas influential. We evaluate our method with a computationally intensive butgold-standard retraining from scratch and demonstrate our methods advantagesover previous methods. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.09418v1 |
|title| VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding |
|authors| Muhammad MaazHanoona RasheedSalman KhanFahad Khan
|links| http://arxiv.org/abs/2406.09418v1 |
|updated| 2024-06-13 17:59:59 UTC |
|summary| Building on the advances of language models Large Multimodal Models LMMshave contributed significant improvements in video understanding. While thecurrent video LMMs utilize advanced Large Language Models LLMs they rely oneither image or video encoders to process visual inputs each of which has itsown limitations. Image encoders excel at capturing rich spatial details fromframe sequences but lack explicit temporal context which can be important invideos with intricate action sequences. On the other hand video encodersprovide temporal context but are often limited by computational constraintsthat lead to processing only sparse frames at lower resolutions resulting inreduced contextual and spatial understanding. To this end we introduceVideoGPT which combines the complementary benefits of the image encoder fordetailed spatial understanding and the video encoder for global temporalcontext modeling. The model processes videos by dividing them into smallersegments and applies an adaptive pooling strategy on features extracted by bothimage and video encoders. Our architecture showcases improved performanceacross multiple video benchmarks including VCGBench MVBench and Zero-shotquestion-answering. Further we develop 112K video-instruction set using anovel semi-automatic annotation pipeline which further improves the modelperformance. Additionally to comprehensively evaluate video LMMs we presentVCGBench-Diverse covering 18 broad video categories such as lifestyle sportsscience gaming and surveillance videos. This benchmark with 4354question-answer pairs evaluates the generalization of existing LMMs on densevideo captioning spatial and temporal understanding and complex reasoningensuring comprehensive assessment across diverse video types and dynamics.Code: https://github.com/mbzuai-oryx/VideoGPT-plus. |


| Item |Content|
| --- |---|
|idx| 2406.09415v1 |
|title| An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels |
|authors| Duy-Kien NguyenMahmoud AssranUnnat JainMartin R. OswaldCees G. M. SnoekXinlei Chen
|links| http://arxiv.org/abs/2406.09415v1 |
|updated| 2024-06-13 17:59:58 UTC |
|summary| This work does not introduce a new method. Instead we present an interestingfinding that questions the necessity of the inductive bias -- locality inmodern computer vision architectures. Concretely we find that vanillaTransformers can operate by directly treating each individual pixel as a tokenand achieve highly performant results. This is substantially different from thepopular design in Vision Transformer which maintains the inductive bias fromConvNets towards local neighborhoods e.g. by treating each 16x16 patch as atoken. We mainly showcase the effectiveness of pixels-as-tokens across threewell-studied tasks in computer vision: supervised learning for objectclassification self-supervised learning via masked autoencoding and imagegeneration with diffusion models. Although directly operating on individualpixels is less computationally practical we believe the community must beaware of this surprising piece of knowledge when devising the next generationof neural architectures for computer vision. |


| Item |Content|
| --- |---|
|idx| 2406.09416v1 |
|title| Alleviating Distortion in Image Generation via Multi-Resolution Diffusion Models |
|authors| Qihao LiuZhanpeng ZengJu HeQihang YuXiaohui ShenLiang-Chieh Chen
|links| http://arxiv.org/abs/2406.09416v1 |
|updated| 2024-06-13 17:59:58 UTC |
|summary| This paper presents innovative enhancements to diffusion models byintegrating a novel multi-resolution network and time-dependent layernormalization. Diffusion models have gained prominence for their effectivenessin high-fidelity image generation. While conventional approaches rely onconvolutional U-Net architectures recent Transformer-based designs havedemonstrated superior performance and scalability. However Transformerarchitectures which tokenize input data via patchification face atrade-off between visual fidelity and computational complexity due to thequadratic nature of self-attention operations concerning token length. Whilelarger patch sizes enable attention computation efficiency they struggle tocapture fine-grained visual details leading to image distortions. To addressthis challenge we propose augmenting the Diffusion model with theMulti-Resolution network DiMR a framework that refines features acrossmultiple resolutions progressively enhancing detail from low to highresolution. Additionally we introduce Time-Dependent Layer NormalizationTD-LN a parameter-efficient approach that incorporates time-dependentparameters into layer normalization to inject time information and achievesuperior performance. Our methods efficacy is demonstrated on theclass-conditional ImageNet generation benchmark where DiMR-XL variantsoutperform prior diffusion models setting new state-of-the-art FID scores of1.70 on ImageNet 256 x 256 and 2.89 on ImageNet 512 x 512. Project page:https://qihao067.github.io/projects/DiMR |


| Item |Content|
| --- |---|
|idx| 2406.09417v1 |
|title| Rethinking Score Distillation as a Bridge Between Image Distributions |
|authors| David McAllisterSongwei GeJia-Bin HuangDavid W. JacobsAlexei A. EfrosAleksander HolynskiAngjoo Kanazawa
|links| http://arxiv.org/abs/2406.09417v1 |
|updated| 2024-06-13 17:59:58 UTC |
|summary| Score distillation sampling SDS has proven to be an important toolenabling the use of large-scale diffusion priors for tasks operating indata-poor domains. Unfortunately SDS has a number of characteristic artifactsthat limit its usefulness in general-purpose applications. In this paper wemake progress toward understanding the behavior of SDS and its variants byviewing them as solving an optimal-cost transport path from a sourcedistribution to a target distribution. Under this new interpretation thesemethods seek to transport corrupted images source to the natural imagedistribution target. We argue that current methods characteristic artifactsare caused by 1 linear approximation of the optimal path and 2 poorestimates of the source distribution. We show that calibrating the textconditioning of the source distribution can produce high-quality generation andtranslation results with little extra overhead. Our method can be easilyapplied across many domains matching or beating the performance of specializedmethods. We demonstrate its utility in text-to-2D text-based NeRFoptimization translating paintings to real images optical illusiongeneration and 3D sketch-to-real. We compare our method to existing approachesfor score distillation sampling and show that it can produce high-frequencydetails with realistic colors. |


| Item |Content|
| --- |---|
|idx| 2406.09413v1 |
|title| Interpreting the Weight Space of Customized Diffusion Models |
|authors| Amil DravidYossi GandelsmanKuan-Chieh WangRameen AbdalGordon WetzsteinAlexei A. EfrosKfir Aberman
|links| http://arxiv.org/abs/2406.09413v1 |
|updated| 2024-06-13 17:59:56 UTC |
|summary| We investigate the space of weights spanned by a large collection ofcustomized diffusion models. We populate this space by creating a dataset ofover 60000 models each of which is a base model fine-tuned to insert adifferent persons visual identity. We model the underlying manifold of theseweights as a subspace which we term weights2weights. We demonstrate threeimmediate applications of this space -- sampling editing and inversion.First as each point in the space corresponds to an identity sampling a set ofweights from it results in a model encoding a novel identity. Next we findlinear directions in this space corresponding to semantic edits of the identitye.g. adding a beard. These edits persist in appearance across generatedsamples. Finally we show that inverting a single image into this spacereconstructs a realistic identity even if the input image is out ofdistribution e.g. a painting. Our results indicate that the weight space offine-tuned diffusion models behaves as an interpretable latent space ofidentities. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.09405v1 |
|title| Why Warmup the Learning Rate? Underlying Mechanisms and Improvements |
|authors| Dayal Singh KalraMaissam Barkeshli
|links| http://arxiv.org/abs/2406.09405v1 |
|updated| 2024-06-13 17:59:35 UTC |
|summary| It is common in deep learning to warm up the learning rate eta often by alinear schedule between eta_textinit  0 and a predetermined targeteta_texttrgt. In this paper we show through systematic experimentsusing SGD and Adam that the overwhelming benefit of warmup arises from allowingthe network to tolerate larger eta_texttrgt by forcing the network tomore well-conditioned areas of the loss landscape. The ability to handle largereta_texttrgt makes hyperparameter tuning more robust while improvingthe final performance. We uncover different regimes of operation during thewarmup period depending on whether training starts off in a progressivesharpening or sharpness reduction phase which in turn depends on theinitialization and parameterization. Using these insights we show howeta_textinit can be properly chosen by utilizing the loss catapultmechanism which saves on the number of warmup steps in some cases completelyeliminating the need for warmup. We also suggest an initialization for thevariance in Adam which provides benefits similar to warmup. |


| Item |Content|
| --- |---|
|idx| 2406.09387v1 |
|title| Oblivious subspace embeddings for compressed Tucker decompositions |
|authors| Matthew PietrosanuBei JiangLinglong Kong
|links| http://arxiv.org/abs/2406.09387v1 |
|updated| 2024-06-13 17:58:32 UTC |
|summary| Emphasis in the tensor literature on random embeddings tools forlow-distortion dimension reduction for the canonical polyadic CP tensordecomposition has left analogous results for the more expressive Tuckerdecomposition comparatively lacking. This work establishes generalJohnson-Lindenstrauss JL type guarantees for the estimation of Tuckerdecompositions when an oblivious random embedding is applied along each mode.When these embeddings are drawn from a JL-optimal family the decomposition canbe estimated within varepsilon relative error under restrictions on theembedding dimension that are in line with recent CP results. We implement ahigher-order orthogonal iteration HOOI decomposition algorithm with randomembeddings to demonstrate the practical benefits of this approach and itspotential to improve the accessibility of otherwise prohibitive tensoranalyses. On moderately large face image and fMRI neuroimaging datasetsempirical results show that substantial dimension reduction is possible withminimal increase in reconstruction error relative to traditional HOOI leq5larger error 50-60 lower computation time for large models with 50dimension reduction along each mode. Especially for large tensors our methodoutperforms traditional higher-order singular value decomposition HOSVD andrecently proposed TensorSketch methods. |


| Item |Content|
| --- |---|
|idx| 2406.09375v1 |
|title| Learning conditional distributions on continuous spaces |
|authors| Cyril BénézetZiteng ChengSebastian Jaimungal
|links| http://arxiv.org/abs/2406.09375v1 |
|updated| 2024-06-13 17:53:47 UTC |
|summary| We investigate sample-based learning of conditional distributions onmulti-dimensional unit boxes allowing for different dimensions of the featureand target spaces. Our approach involves clustering data near varying querypoints in the feature space to create empirical measures in the target space.We employ two distinct clustering schemes: one based on a fixed-radius ball andthe other on nearest neighbors. We establish upper bounds for the convergencerates of both methods and from these bounds deduce optimal configurations forthe radius and the number of neighbors. We propose to incorporate the nearestneighbors method into neural network training as our empirical analysisindicates it has better performance in practice. For efficiency our trainingprocess utilizes approximate nearest neighbors search with random binary spacepartitioning. Additionally we employ the Sinkhorn algorithm and asparsity-enforced transport plan. Our empirical findings demonstrate that witha suitably designed structure the neural network has the ability to adapt to asuitable level of Lipschitz continuity locally. For reproducibility our codeis available at urlhttps://github.com/zcheng-a/LCD_kNN. |


| Item |Content|
| --- |---|
|idx| 2406.09357v1 |
|title| Advancing Graph Generation through Beta Diffusion |
|authors| Yilin HeXinyang LiuBo ChenMingyuan Zhou
|links| http://arxiv.org/abs/2406.09357v1 |
|updated| 2024-06-13 17:42:57 UTC |
|summary| Diffusion models have demonstrated effectiveness in generating natural imagesand have been extended to generate diverse data types including graphs. Thisnew generation of diffusion-based graph generative models has demonstratedsignificant performance improvements over methods that rely on variationalautoencoders or generative adversarial networks. Its important to recognizehowever that most of these models employ Gaussian or categorical diffusionprocesses which can struggle with sparse and long-tailed data distributions.In our work we introduce Graph Beta Diffusion GBD a diffusion-basedgenerative model particularly adept at capturing diverse graph structures. GBDutilizes a beta diffusion process tailored for the sparse and range-boundedcharacteristics of graph adjacency matrices. Furthermore we have developed amodulation technique that enhances the realism of the generated graphs bystabilizing the generation of critical graph structures while preservingflexibility elsewhere. The outstanding performance of GBD across three generalgraph benchmarks and two biochemical graph benchmarks highlights its capabilityto effectively capture the complexities of real-world graph data. The code willbe made available at https://github.com/YH-UtMSB/Graph_Beta_Diffusion |


| Item |Content|
| --- |---|
|idx| 2406.09347v1 |
|title| Separations in the Representational Capabilities of Transformers and Recurrent Architectures |
|authors| Satwik BhattamishraMichael HahnPhil BlunsomVarun Kanade
|links| http://arxiv.org/abs/2406.09347v1 |
|updated| 2024-06-13 17:31:30 UTC |
|summary| Transformer architectures have been widely adopted in foundation models. Dueto their high inference costs there is renewed interest in exploring thepotential of efficient recurrent architectures RNNs. In this paper weanalyze the differences in the representational capabilities of Transformersand RNNs across several tasks of practical relevance including index lookupnearest neighbor recognizing bounded Dyck languages and string equality. Forthe tasks considered our results show separations based on the size of themodel required for different architectures. For example we show that aone-layer Transformer of logarithmic width can perform index lookup whereas anRNN requires a hidden state of linear size. Conversely while constant-sizeRNNs can recognize bounded Dyck languages we show that one-layer Transformersrequire a linear size for this task. Furthermore we show that two-layerTransformers of logarithmic size can perform decision tasks such as stringequality or disjointness whereas both one-layer Transformers and recurrentmodels require linear size for these tasks. We also show that a log-sizetwo-layer Transformer can implement the nearest neighbor algorithm in itsforward pass on the other hand recurrent models require linear size. Ourconstructions are based on the existence of N nearly orthogonal vectors inOlog N dimensional space and our lower bounds are based on reductions fromcommunication complexity problems. We supplement our theoretical results withexperiments that highlight the differences in the performance of thesearchitectures on practical-size sequences. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.09264v1 |
|title| Towards Bidirectional Human-AI Alignment: A Systematic Review for Clarifications, Framework, and Future Directions |
|authors| Hua ShenTiffany KnearemReshmi GhoshKenan AlkiekKundan KrishnaYachuan LiuZiqiao MaSavvas PetridisYi-Hao PengLi QiweiSushrita RakshitChenglei SiYutong XieJeffrey P. BighamFrank BentleyJoyce ChaiZachary LiptonQiaozhu MeiRada MihalceaMichael TerryDiyi YangMeredith Ringel MorrisPaul ResnickDavid Jurgens
|links| http://arxiv.org/abs/2406.09264v1 |
|updated| 2024-06-13 16:03:25 UTC |
|summary| Recent advancements in general-purpose AI have highlighted the importance ofguiding AI systems towards the intended goals ethical principles and valuesof individuals and groups a concept broadly recognized as alignment. Howeverthe lack of clarified definitions and scopes of human-AI alignment poses asignificant obstacle hampering collaborative efforts across research domainsto achieve this alignment. In particular ML- and philosophy-oriented alignmentresearch often views AI alignment as a static unidirectional process i.e.aiming to ensure that AI systems objectives match humans rather than anongoing mutual alignment problem 429. This perspective largely neglects thelong-term interaction and dynamic changes of alignment. To understand thesegaps we introduce a systematic review of over 400 papers published between2019 and January 2024 spanning multiple domains such as Human-ComputerInteraction HCI Natural Language Processing NLP Machine Learning MLand others. We characterize define and scope human-AI alignment. From this wepresent a conceptual framework of Bidirectional Human-AI Alignment toorganize the literature from a human-centered perspective. This frameworkencompasses both 1 conventional studies of aligning AI to humans that ensuresAI produces the intended outcomes determined by humans and 2 a proposedconcept of aligning humans to AI which aims to help individuals and societyadjust to AI advancements both cognitively and behaviorally. Additionally wearticulate the key findings derived from literature analysis includingdiscussions about human values interaction techniques and evaluations. Topave the way for future studies we envision three key challenges for futuredirections and propose examples of potential future solutions. |


| Item |Content|
| --- |---|
|idx| 2406.09037v1 |
|title| Evaluating Privacy, Security, and Trust Perceptions in Conversational AI: A Systematic Review |
|authors| Anna LeschanowskySilas RechBirgit PoppTom Bäckström
|links| http://arxiv.org/abs/2406.09037v1 |
|updated| 2024-06-13 12:20:26 UTC |
|summary| Conversational AI CAI systems which encompass voice- and text-basedassistants are on the rise and have been largely integrated into peopleseveryday lives. Despite their widespread adoption users voice concernsregarding privacy security and trust in these systems. However thecomposition of these perceptions their impact on technology adoption and usageand the relationship between privacy security and trust perceptions in the CAIcontext remain open research challenges. This study contributes to the field byconducting a Systematic Literature Review and offers insights into the currentstate of research on privacy security and trust perceptions in the context ofCAI systems. The review covers application fields and user groups and shedslight on empirical methods and tools used for assessment. Moreover it providesinsights into the reliability and validity of privacy security and trustscales as well as extensively investigating the subconstructs of each item aswell as additional concepts which are concurrently collected. We point out thatthe perceptions of trust privacy and security overlap based on thesubconstructs we identified. While the majority of studies investigate one ofthese concepts only a few studies were found exploring privacy security andtrust perceptions jointly. Our research aims to inform on directions to developand use reliable scales for users privacy security and trust perceptions andcontribute to the development of trustworthy CAI systems. |


| Item |Content|
| --- |---|
|idx| 2406.08959v1 |
|title| Beyond Recommendations: From Backward to Forward AI Support of Pilots' Decision-Making Process |
|authors| Zelun Tony ZhangSebastian S. FegerLucas DullenkopfRulu LiaoLukas SüsslinYuanting LiuAndreas Butz
|links| http://arxiv.org/abs/2406.08959v1 |
|updated| 2024-06-13 09:44:04 UTC |
|summary| AI is anticipated to enhance human decision-making in high-stakes domainslike aviation but adoption is often hindered by challenges such asinappropriate reliance and poor alignment with users decision-making. Recentresearch suggests that a core underlying issue is the recommendation-centricdesign of many AI systems i.e. they give end-to-end recommendations andignore the rest of the decision-making process. Alternative support paradigmsare rare and it remains unclear how the few that do exist compare torecommendation-centric support. In this work we aimed to empirically comparerecommendation-centric support to an alternative paradigm continuous supportin the context of diversions in aviation. We conducted a mixed-methods studywith 32 professional pilots in a realistic setting. To ensure the quality ofour study scenarios we conducted a focus group with four additional pilotsprior to the study. We found that continuous support can support pilotsdecision-making in a forward direction allowing them to think more beyond thelimits of the system and make faster decisions when combined withrecommendations though the forward support can be disrupted. Participantsstatements further suggest a shift in design goal away from providingrecommendations to supporting quick information gathering. Our results showways to design more helpful and effective AI decision support that goes beyondend-to-end recommendations. |


| Item |Content|
| --- |---|
|idx| 2406.08946v1 |
|title| Human-Robot Interface for Teleoperated Robotized Planetary Sample Collection and Assembly |
|authors| Lorenzo PagliaraVincenzo PetroneEnrico FerrentinoPasquale Chiacchio
|links| http://dx.doi.org/10.1109/MetroAeroSpace57412.2023.10189984 |
|updated| 2024-06-13 09:17:10 UTC |
|summary| As human space exploration evolves toward longer voyages farther from ourhome planet in-situ resource utilization ISRU becomes increasinglyimportant. Haptic teleoperations are one of the technologies by which suchactivities can be carried out remotely by humans whose expertise is stillnecessary for complex activities. In order to perform precision tasks witheffectiveness the operator must experience ease of use and accuracy. The samefeatures are demanded to reduce the complexity of the training procedures andthe associated learning time for operators without a specific background inrobotic teleoperations. Haptic teleoperation systems that allow for a naturalfeeling of forces need to cope with the trade-off between accurate movementsand workspace extension. Clearly both of them are required for typical ISRUtasks. In this work we develop a new concept of operations and suitablehuman-robot interfaces to achieve sample collection and assembly with ease ofuse and accuracy. In the proposed operational concept the teleoperation spaceis extended by executing automated trajectories offline planned at the controlstation. In three different experimental scenarios we validate the end-to-endsystem involving the control station and the robotic asset by assessing thecontribution of haptics to mission success the system robustness to consistentdelays and the ease of training new operators. |


| Item |Content|
| --- |---|
|idx| 2406.08875v1 |
|title| NICER: A New and Improved Consumed Endurance and Recovery Metric to Quantify Muscle Fatigue of Mid-Air Interactions |
|authors| Yi LiBenjamin TagShaozhang DaiRobert CrowtherTim DwyerPourang IraniBarrett Ens
|links| http://dx.doi.org/10.1145/3658230 |
|updated| 2024-06-13 07:22:48 UTC |
|summary| Natural gestures are crucial for mid-air interaction but predicting andmanaging muscle fatigue is challenging. Existing torque-based models arelimited in their ability to model above-shoulder interactions and to accountfor fatigue recovery. We introduce a new hybrid model NICER which combines atorque-based approach with a new term derived from the empirical measurement ofmuscle contraction and a recovery factor to account for decreasing fatigueduring rest. We evaluated NICER in a mid-air selection task using twointeraction methods with different degrees of perceived fatigue. Results showthat NICER can accurately model above-shoulder interactions as well as reflectfatigue recovery during rest periods. Moreover both interaction methods show astronger correlation with subjective fatigue measurement r  0.978/0.976 thana previous model Cumulative Fatigue r  0.966/ 0.923 confirming that NICERis a powerful analytical tool to predict fatigue across a variety ofgesture-based interactive applications. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.09318v1 |
|title| Characterising Interventions in Causal Games |
|authors| Manuj MishraJames FoxMichael Wooldridge
|links| http://arxiv.org/abs/2406.09318v1 |
|updated| 2024-06-13 16:55:07 UTC |
|summary| Causal games are probabilistic graphical models that enable causal queries tobe answered in multi-agent settings. They extend causal Bayesian networks byspecifying decision and utility variables to represent the agents degrees offreedom and objectives. In multi-agent settings whether each agent decides ontheir policy before or after knowing the causal intervention is important asthis affects whether they can respond to the intervention by adapting theirpolicy. Consequently previous work in causal games imposed chronologicalconstraints on permissible interventions. We relax this by outlining a soundand complete set of primitive causal interventions so the effect of anyarbitrarily complex interventional query can be studied in multi-agentsettings. We also demonstrate applications to the design of safe AI systems byconsidering causal mechanism design and commitment. |


| Item |Content|
| --- |---|
|idx| 2406.09214v1 |
|title| Applying Multi-Agent Negotiation to Solve the Production Routing Problem With Privacy Preserving |
|authors| Luiza Pellin BiasotoVinicius Renan de CarvalhoJaime Simão Sichman
|links| http://arxiv.org/abs/2406.09214v1 |
|updated| 2024-06-13 15:15:34 UTC |
|summary| This paper presents a novel approach to address the Production RoutingProblem with Privacy Preserving PRPPP in supply chain optimization. Theintegrated optimization of production inventory distribution and routingdecisions in real-world industry applications poses several challengesincluding increased complexity discrepancies between planning and executionand constraints on information sharing. To mitigate these challenges thispaper proposes the use of intelligent agent negotiation within a hybridMulti-Agent System MAS integrated with optimization algorithms. The MASfacilitates communication and coordination among entities encapsulates privateinformation and enables negotiation. This along with optimization algorithmsmakes it a compelling framework for establishing optimal solutions. Theapproach is supported by real-world applications and synergies between MAS andoptimization methods demonstrating its effectiveness in addressing complexsupply chain optimization problems. |


| Item |Content|
| --- |---|
|idx| 2406.08979v1 |
|title| Multi-Agent Software Development through Cross-Team Collaboration |
|authors| Zhuoyun DuChen QianWei LiuZihao XieYifei WangYufan DangWeize ChenCheng Yang
|links| http://arxiv.org/abs/2406.08979v1 |
|updated| 2024-06-13 10:18:36 UTC |
|summary| The latest breakthroughs in Large Language Models LLMs eg. ChatDev havecatalyzed profound transformations particularly through multi-agentcollaboration for software development. LLM agents can collaborate in teamslike humans and follow the waterfall model to sequentially work onrequirements analysis development review testing and other phases toperform autonomous software generation. However for an agent team each phasein a single development process yields only one possible outcome. This resultsin the completion of only one development chain thereby losing the opportunityto explore multiple potential decision paths within the solution space.Consequently this may lead to obtaining suboptimal results. To address thischallenge we introduce Cross-Team Collaboration CTC a scalable multi-teamframework that enables orchestrated teams to jointly propose various decisionsand communicate with their insights in a cross-team collaboration environmentfor superior content generation. Experimental results in software developmentreveal a notable increase in quality compared to state-of-the-art baselinesunderscoring the efficacy of our framework. The significant improvements instory generation demonstrate the promising generalization ability of ourframework across various domains. We anticipate that our work will guide LLMagents towards a cross-team paradigm and contribute to their significant growthin but not limited to software development. The code and data will be availableat https://github.com/OpenBMB/ChatDev. |


| Item |Content|
| --- |---|
|idx| 2406.08440v1 |
|title| Adaptive Swarm Mesh Refinement using Deep Reinforcement Learning with Local Rewards |
|authors| Niklas FreymuthPhilipp DahlingerTobias WürthSimon ReischLuise KärgerGerhard Neumann
|links| http://arxiv.org/abs/2406.08440v1 |
|updated| 2024-06-12 17:26:54 UTC |
|summary| Simulating physical systems is essential in engineering but analyticalsolutions are limited to straightforward problems. Consequently numericalmethods like the Finite Element Method FEM are widely used. However the FEMbecomes computationally expensive as problem complexity and accuracy demandsincrease. Adaptive Mesh Refinement AMR improves the FEM by dynamicallyallocating mesh elements on the domain balancing computational speed andaccuracy. Classical AMR depends on heuristics or expensive error estimatorslimiting its use in complex simulations. While learning-based AMR methods arepromising they currently only scale to simple problems. In this work weformulate AMR as a system of collaborating homogeneous agents that iterativelysplit into multiple new agents. This agent-wise perspective enables a spatialreward formulation focused on reducing the maximum mesh element error. Ourapproach Adaptive Swarm Mesh Refinement ASMR offers efficient stableoptimization and generates highly adaptive meshes at user-defined resolutionduring inference. Extensive experiments including volumetric meshes andNeumann boundary conditions demonstrate that ASMR exceeds heuristic approachesand learned baselines matching the performance of expensive error-based oracleAMR strategies. ASMR additionally generalizes to different domains duringinference and produces meshes that simulate up to 2 orders of magnitude fasterthan uniform refinements in more demanding settings. |


| Item |Content|
| --- |---|
|idx| 2406.08002v1 |
|title| Efficient Adaptation in Mixed-Motive Environments via Hierarchical Opponent Modeling and Planning |
|authors| Yizhe HuangAnji LiuFanqi KongYaodong YangSong-Chun ZhuXue Feng
|links| http://arxiv.org/abs/2406.08002v1 |
|updated| 2024-06-12 08:48:06 UTC |
|summary| Despite the recent successes of multi-agent reinforcement learning MARLalgorithms efficiently adapting to co-players in mixed-motive environmentsremains a significant challenge. One feasible approach is to hierarchicallymodel co-players behavior based on inferring their characteristics. Howeverthese methods often encounter difficulties in efficient reasoning andutilization of inferred information. To address these issues we proposeHierarchical Opponent modeling and Planning HOP a novel multi-agentdecision-making algorithm that enables few-shot adaptation to unseen policiesin mixed-motive environments. HOP is hierarchically composed of two modules: anopponent modeling module that infers others goals and learns correspondinggoal-conditioned policies and a planning module that employs Monte Carlo TreeSearch MCTS to identify the best response. Our approach improves efficiencyby updating beliefs about others goals both across and within episodes and byusing information from the opponent modeling module to guide planning.Experimental results demonstrate that in mixed-motive environments HOPexhibits superior few-shot adaptation capabilities when interacting withvarious unseen agents and excels in self-play scenarios. Furthermore theemergence of social intelligence during our experiments underscores thepotential of our approach in complex multi-agent environments. |


