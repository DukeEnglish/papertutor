# cs.CL 

| Item |Content|
| --- |---|
|idx| 2409.18110v1 |
|title| Open-World Evaluation for Retrieving Diverse Perspectives |
|authors| Hung-Ting ChenEunsol Choi
|links| http://arxiv.org/abs/2409.18110v1 |
|updated| 2024-09-26 17:52:57 UTC |
|summary| We study retrieving a set of documents that covers various perspectives on acomplex and contentious question e.g. will ChatGPT do more harm than good.We curate a Benchmark for Retrieval Diversity for Subjective questions BERDSwhere each example consists of a question and diverse perspectives associatedwith the question sourced from survey questions and debate websites. On thisdata retrievers paired with a corpus are evaluated to surface a document setthat contains diverse perspectives. Our framing diverges from most retrievaltasks in that document relevancy cannot be decided by simple string matches toreferences. Instead we build a language model based automatic evaluator thatdecides whether each retrieved document contains a perspective. This allows usto evaluate the performance of three different types of corpus Wikipedia websnapshot and corpus constructed on the fly with retrieved pages from thesearch engine paired with retrievers. Retrieving diverse documents remainschallenging with the outputs from existing retrievers covering allperspectives on only 33.74 of the examples. We further study the impact ofquery expansion and diversity-focused reranking approaches and analyzeretriever sycophancy. Together we lay the foundation for future studies inretrieval diversity handling complex queries. |


| Item |Content|
| --- |---|
|idx| 2409.18073v1 |
|title| Infer Human's Intentions Before Following Natural Language Instructions |
|authors| Yanming WanYue WuYiping WangJiayuan MaoNatasha Jaques
|links| http://arxiv.org/abs/2409.18073v1 |
|updated| 2024-09-26 17:19:49 UTC |
|summary| For AI agents to be helpful to humans they should be able to follow naturallanguage instructions to complete everyday cooperative tasks in humanenvironments. However real human instructions inherently possess ambiguitybecause the human speakers assume sufficient prior knowledge about their hiddengoals and intentions. Standard language grounding and planning methods fail toaddress such ambiguities because they do not model human internal goals asadditional partially observable factors in the environment. We propose a newframework Follow Instructions with Social and Embodied Reasoning FISERaiming for better natural language instruction following in collaborativeembodied tasks. Our framework makes explicit inferences about human goals andintentions as intermediate reasoning steps. We implement a set ofTransformer-based models and evaluate them over a challenging benchmarkHandMeThat. We empirically demonstrate that using social reasoning toexplicitly infer human intentions before making action plans surpasses purelyend-to-end approaches. We also compare our implementation with strongbaselines including Chain of Thought prompting on the largest availablepre-trained language models and find that FISER provides better performance onthe embodied social reasoning tasks under investigation reaching thestate-of-the-art on HandMeThat. |


| Item |Content|
| --- |---|
|idx| 2409.18046v1 |
|title| IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning |
|authors| Soeun LeeSi-Woo KimTaewhan KimDong-Jin Kim
|links| http://arxiv.org/abs/2409.18046v1 |
|updated| 2024-09-26 16:47:32 UTC |
|summary| Recent advancements in image captioning have explored text-only trainingmethods to overcome the limitations of paired image-text data. Howeverexisting text-only training methods often overlook the modality gap betweenusing text data during training and employing images during inference. Toaddress this issue we propose a novel approach called Image-like Retrievalwhich aligns text features with visually relevant features to mitigate themodality gap. Our method further enhances the accuracy of generated captions bydesigning a Fusion Module that integrates retrieved captions with inputfeatures. Additionally we introduce a Frequency-based Entity Filteringtechnique that significantly improves caption quality. We integrate thesemethods into a unified framework which we refer to as IFCaptextbfImage-like Retrieval and textbfFrequency-based EntityFiltering for Zero-shot textbfCaptioning. Through extensiveexperimentation our straightforward yet powerful approach has demonstrated itsefficacy outperforming the state-of-the-art methods by a significant margin inboth image captioning and video captioning compared to zero-shot captioningbased on text-only training. |


| Item |Content|
| --- |---|
|idx| 2409.18044v1 |
|title| Unveiling the Role of Pretraining in Direct Speech Translation |
|authors| Belen AlastrueyGerard I. GállegoMarta R. Costa-jussà
|links| http://arxiv.org/abs/2409.18044v1 |
|updated| 2024-09-26 16:46:46 UTC |
|summary| Direct speech-to-text translation systems encounter an important drawback indata scarcity. A common solution consists on pretraining the encoder onautomatic speech recognition hence losing efficiency in the training process.In this study we compare the training dynamics of a system using a pretrainedencoder the conventional approach and one trained from scratch. We observethat throughout the training the randomly initialized model struggles toincorporate information from the speech inputs for its predictions. Hence wehypothesize that this issue stems from the difficulty of effectively trainingan encoder for direct speech translation. While a model trained from scratchneeds to learn acoustic and semantic modeling simultaneously a pretrained onecan just focus on the latter. Based on these findings we propose a subtlechange in the decoder cross-attention to integrate source information fromearlier steps in training. We show that with this change the model trainedfrom scratch can achieve comparable performance to the pretrained one whilereducing the training time. |


| Item |Content|
| --- |---|
|idx| 2409.18042v1 |
|title| EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions |
|authors| Kai ChenYunhao GouRunhui HuangZhili LiuDaxin TanJing XuChunwei WangYi ZhuYihan ZengKuo YangDingdong WangKun XiangHaoyuan LiHaoli BaiJianhua HanXiaohui LiWeike JinNian XieYu ZhangJames T. KwokHengshuang ZhaoXiaodan LiangDit-Yan YeungXiao ChenZhenguo LiWei ZhangQun LiuLanqing HongLu HouHang Xu
|links| http://arxiv.org/abs/2409.18042v1 |
|updated| 2024-09-26 16:44:02 UTC |
|summary| GPT-4o an omni-modal model that enables vocal conversations with diverseemotions and tones marks a milestone for omni-modal foundation models.However empowering Large Language Models to perceive and generate imagestexts and speeches end-to-end with publicly available data remains challengingin the open-source community. Existing vision-language models rely on externaltools for the speech processing while speech-language models still suffer fromlimited or even without vision-understanding abilities. To address this gap wepropose EMOVA EMotionally Omni-present Voice Assistant to enable LargeLanguage Models with end-to-end speech capabilities while maintaining theleading vision-language performance. With a semantic-acoustic disentangledspeech tokenizer we notice surprisingly that omni-modal alignment can furtherenhance vision-language and speech abilities compared with the correspondingbi-modal aligned counterparts. Moreover a lightweight style module is proposedfor flexible speech style controls e.g. emotions and pitches. For the firsttime EMOVA achieves state-of-the-art performance on both the vision-languageand speech benchmarks and meanwhile supporting omni-modal spoken dialoguewith vivid emotions. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2409.18119v1 |
|title| Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography |
|authors| Yuexi DuJohn OnofreyNicha C. Dvornek
|links| http://arxiv.org/abs/2409.18119v1 |
|updated| 2024-09-26 17:56:59 UTC |
|summary| Contrastive Language-Image Pre-training CLIP shows promise in medical imageanalysis but requires substantial data and computational resources. Due tothese restrictions existing CLIP applications in medical imaging focus mainlyon modalities like chest X-rays that have abundant image-report data availableleaving many other important modalities under-explored. Here we propose thefirst adaptation of the full CLIP model to mammography which presentssignificant challenges due to labeled data scarcity high-resolution imageswith small regions of interest and data imbalance. We first develop aspecialized supervision framework for mammography that leverages its multi-viewnature. Furthermore we design a symmetric local alignment module to betterfocus on detailed features in high-resolution images. Lastly we incorporate aparameter-efficient fine-tuning approach for large language models pre-trainedwith medical knowledge to address data limitations. Our multi-view andmulti-scale alignment MaMA method outperforms state-of-the-art baselines forthree different tasks on two large real-world mammography datasets EMBED andRSNA-Mammo with only 52 model size compared with the largest baseline. |


| Item |Content|
| --- |---|
|idx| 2409.18104v1 |
|title| Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats |
|authors| Lucia GordonNikhil BehariSamuel CollierElizabeth Bondi-KellyJackson A. KillianCatherine RessijacPeter BoucherAndrew DaviesMilind Tambe
|links| http://dx.doi.org/10.24963/ijcai.2023/663 |
|updated| 2024-09-26 17:49:20 UTC |
|summary| Much of Earths charismatic megafauna is endangered by human activitiesparticularly the rhino which is at risk of extinction due to the poachingcrisis in Africa. Monitoring rhinos movement is crucial to their protectionbut has unfortunately proven difficult because rhinos are elusive. Thereforeinstead of tracking rhinos we propose the novel approach of mapping communaldefecation sites called middens which give information about rhinos spatialbehavior valuable to anti-poaching management and reintroduction efforts.This paper provides the first-ever mapping of rhino midden locations bybuilding classifiers to detect them using remotely sensed thermal RGB andLiDAR imagery in passive and active learning settings. As existing activelearning methods perform poorly due to the extreme class imbalance in ourdataset we design MultimodAL an active learning system employing a rankingtechnique and multimodality to achieve competitive performance with passivelearning models with 94 fewer labels. Our methods could therefore save over 76hours in labeling time when used on a similarly-sized dataset. Unexpectedlyour midden map reveals that rhino middens are not randomly distributedthroughout the landscape rather they are clustered. Consequently rangersshould be targeted at areas with high midden densities to strengthenanti-poaching efforts in line with UN Target 15.7. |


| Item |Content|
| --- |---|
|idx| 2409.18101v1 |
|title| AI-Powered Augmented Reality for Satellite Assembly, Integration and Test |
|authors| Alvaro PatricioJoao ValenteAtabak DehbanInes CadilhaDaniel ReisRodrigo Ventura
|links| http://arxiv.org/abs/2409.18101v1 |
|updated| 2024-09-26 17:44:52 UTC |
|summary| The integration of Artificial Intelligence AI and Augmented Reality AR isset to transform satellite Assembly Integration and Testing AIT processesby enhancing precision minimizing human error and improving operationalefficiency in cleanroom environments. This paper presents a technicaldescription of the European Space Agencys ESA project AI for AR inSatellite AIT which combines real-time computer vision and AR systems toassist technicians during satellite assembly. Leveraging Microsoft HoloLens 2as the AR interface the system delivers context-aware instructions andreal-time feedback tackling the complexities of object recognition and 6D poseestimation in AIT workflows. All AI models demonstrated over 70 accuracy withthe detection model exceeding 95 accuracy indicating a high level ofperformance and reliability. A key contribution of this work lies in theeffective use of synthetic data for training AI models in AR applicationsaddressing the significant challenges of obtaining real-world datasets inhighly dynamic satellite environments as well as the creation of the SegmentedAnything Model for Automatic Labelling SAMAL which facilitates the automaticannotation of real data achieving speeds up to 20 times faster than manualhuman annotation. The findings demonstrate the efficacy of AI-driven AR systemsin automating critical satellite assembly tasks setting a foundation forfuture innovations in the space industry. |


| Item |Content|
| --- |---|
|idx| 2409.18099v1 |
|title| EfficientCrackNet: A Lightweight Model for Crack Segmentation |
|authors| Abid Hasan ZimAquib IqbalZaid Al-HudaAsad MalikMinoru Kuribayash
|links| http://arxiv.org/abs/2409.18099v1 |
|updated| 2024-09-26 17:44:20 UTC |
|summary| Crack detection particularly from pavement images presents a formidablechallenge in the domain of computer vision due to several inherent complexitiessuch as intensity inhomogeneity intricate topologies low contrast and noisybackgrounds. Automated crack detection is crucial for maintaining thestructural integrity of essential infrastructures including buildingspavements and bridges. Existing lightweight methods often face challengesincluding computational inefficiency complex crack patterns and difficultbackgrounds leading to inaccurate detection and impracticality for real-worldapplications. To address these limitations we propose EfficientCrackNet alightweight hybrid model combining Convolutional Neural Networks CNNs andtransformers for precise crack segmentation. EfficientCrackNet integratesdepthwise separable convolutions DSC layers and MobileViT block to captureboth global and local features. The model employs an Edge Extraction MethodEEM and for efficient crack edge detection without pretraining andUltra-Lightweight Subspace Attention Module ULSAM to enhance featureextraction. Extensive experiments on three benchmark datasets Crack500DeepCrack and GAPs384 demonstrate that EfficientCrackNet achieves superiorperformance compared to existing lightweight models while requiring only 0.26Mparameters and 0.483 FLOPs G. The proposed model offers an optimal balancebetween accuracy and computational efficiency outperforming state-of-the-artlightweight models and providing a robust and adaptable solution forreal-world crack segmentation. |


| Item |Content|
| --- |---|
|idx| 2409.18092v1 |
|title| DiffSSC: Semantic LiDAR Scan Completion using Denoising Diffusion Probabilistic Models |
|authors| Helin CaoSven Behnke
|links| http://arxiv.org/abs/2409.18092v1 |
|updated| 2024-09-26 17:39:05 UTC |
|summary| Perception systems play a crucial role in autonomous driving incorporatingmultiple sensors and corresponding computer vision algorithms. 3D LiDAR sensorsare widely used to capture sparse point clouds of the vehicles surroundings.However such systems struggle to perceive occluded areas and gaps in the scenedue to the sparsity of these point clouds and their lack of semantics. Toaddress these challenges Semantic Scene Completion SSC jointly predictsunobserved geometry and semantics in the scene given raw LiDAR measurementsaiming for a more complete scene representation. Building on promising resultsof diffusion models in image generation and super-resolution tasks we proposetheir extension to SSC by implementing the noising and denoising diffusionprocesses in the point and semantic spaces individually. To control thegeneration we employ semantic LiDAR point clouds as conditional input anddesign local and global regularization losses to stabilize the denoisingprocess. We evaluate our approach on autonomous driving datasets and ourapproach outperforms the state-of-the-art for SSC. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2409.18119v1 |
|title| Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography |
|authors| Yuexi DuJohn OnofreyNicha C. Dvornek
|links| http://arxiv.org/abs/2409.18119v1 |
|updated| 2024-09-26 17:56:59 UTC |
|summary| Contrastive Language-Image Pre-training CLIP shows promise in medical imageanalysis but requires substantial data and computational resources. Due tothese restrictions existing CLIP applications in medical imaging focus mainlyon modalities like chest X-rays that have abundant image-report data availableleaving many other important modalities under-explored. Here we propose thefirst adaptation of the full CLIP model to mammography which presentssignificant challenges due to labeled data scarcity high-resolution imageswith small regions of interest and data imbalance. We first develop aspecialized supervision framework for mammography that leverages its multi-viewnature. Furthermore we design a symmetric local alignment module to betterfocus on detailed features in high-resolution images. Lastly we incorporate aparameter-efficient fine-tuning approach for large language models pre-trainedwith medical knowledge to address data limitations. Our multi-view andmulti-scale alignment MaMA method outperforms state-of-the-art baselines forthree different tasks on two large real-world mammography datasets EMBED andRSNA-Mammo with only 52 model size compared with the largest baseline. |


| Item |Content|
| --- |---|
|idx| 2409.18104v1 |
|title| Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats |
|authors| Lucia GordonNikhil BehariSamuel CollierElizabeth Bondi-KellyJackson A. KillianCatherine RessijacPeter BoucherAndrew DaviesMilind Tambe
|links| http://dx.doi.org/10.24963/ijcai.2023/663 |
|updated| 2024-09-26 17:49:20 UTC |
|summary| Much of Earths charismatic megafauna is endangered by human activitiesparticularly the rhino which is at risk of extinction due to the poachingcrisis in Africa. Monitoring rhinos movement is crucial to their protectionbut has unfortunately proven difficult because rhinos are elusive. Thereforeinstead of tracking rhinos we propose the novel approach of mapping communaldefecation sites called middens which give information about rhinos spatialbehavior valuable to anti-poaching management and reintroduction efforts.This paper provides the first-ever mapping of rhino midden locations bybuilding classifiers to detect them using remotely sensed thermal RGB andLiDAR imagery in passive and active learning settings. As existing activelearning methods perform poorly due to the extreme class imbalance in ourdataset we design MultimodAL an active learning system employing a rankingtechnique and multimodality to achieve competitive performance with passivelearning models with 94 fewer labels. Our methods could therefore save over 76hours in labeling time when used on a similarly-sized dataset. Unexpectedlyour midden map reveals that rhino middens are not randomly distributedthroughout the landscape rather they are clustered. Consequently rangersshould be targeted at areas with high midden densities to strengthenanti-poaching efforts in line with UN Target 15.7. |


| Item |Content|
| --- |---|
|idx| 2409.18102v1 |
|title| MALPOLON: A Framework for Deep Species Distribution Modeling |
|authors| Theo LarcherLukas PicekBenjamin DeneuTitouan LorieulMaximilien ServajeanAlexis Joly
|links| http://arxiv.org/abs/2409.18102v1 |
|updated| 2024-09-26 17:45:10 UTC |
|summary| This paper describes a deep-SDM framework MALPOLON. Written in Python andbuilt upon the PyTorch library this framework aims to facilitate training andinferences of deep species distribution models deep-SDM and sharing for userswith only general Python language skills e.g. modeling ecologists who areinterested in testing deep learning approaches to build new SDMs. More advancedusers can also benefit from the frameworks modularity to run more specificexperiments by overriding existing classes while taking advantage ofpress-button examples to train neural networks on multiple classification tasksusing custom or provided raw and pre-processed datasets. The framework isopen-sourced on GitHub and PyPi along with extensive documentation and examplesof use in various scenarios. MALPOLON offers straightforward installationYAML-based configuration parallel computing multi-GPU utilization baselineand foundational models for benchmarking and extensivetutorials/documentation aiming to enhance accessibility and performancescalability for ecologists and researchers. |


| Item |Content|
| --- |---|
|idx| 2409.18100v1 |
|title| Self-supervised Pretraining for Cardiovascular Magnetic Resonance Cine Segmentation |
|authors| Rob A. J. de MooijJosien P. W. PluimCian M. Scannell
|links| http://arxiv.org/abs/2409.18100v1 |
|updated| 2024-09-26 17:44:29 UTC |
|summary| Self-supervised pretraining SSP has shown promising results in learningfrom large unlabeled datasets and thus could be useful for automatedcardiovascular magnetic resonance CMR short-axis cine segmentation. Howeverinconsistent reports of the benefits of SSP for segmentation have made itdifficult to apply SSP to CMR. Therefore this study aimed to evaluate SSPmethods for CMR cine segmentation.  To this end short-axis cine stacks of 296 subjects 90618 2D slices wereused for unlabeled pretraining with four SSP methods SimCLR positionalcontrastive learning DINO and masked image modeling MIM. Subsets of varyingnumbers of subjects were used for supervised fine-tuning of 2D models for eachSSP method as well as to train a 2D baseline model from scratch. Thefine-tuned models were compared to the baseline using the 3D Dice similaritycoefficient DSC in a test dataset of 140 subjects.  The SSP methods showed no performance gains with the largest supervisedfine-tuning subset compared to the baseline DSC  0.89. When only 10 subjects231 2D slices are available for supervised training SSP using MIM DSC 0.86 improves over training from scratch DSC  0.82.  This study found that SSP is valuable for CMR cine segmentation when labeledtraining data is scarce but does not aid state-of-the-art deep learningmethods when ample labeled data is available. Moreover the choice of SSPmethod is important. The code is publicly available at:https://github.com/q-cardIA/ssp-cmr-cine-segmentation |


| Item |Content|
| --- |---|
|idx| 2409.18073v1 |
|title| Infer Human's Intentions Before Following Natural Language Instructions |
|authors| Yanming WanYue WuYiping WangJiayuan MaoNatasha Jaques
|links| http://arxiv.org/abs/2409.18073v1 |
|updated| 2024-09-26 17:19:49 UTC |
|summary| For AI agents to be helpful to humans they should be able to follow naturallanguage instructions to complete everyday cooperative tasks in humanenvironments. However real human instructions inherently possess ambiguitybecause the human speakers assume sufficient prior knowledge about their hiddengoals and intentions. Standard language grounding and planning methods fail toaddress such ambiguities because they do not model human internal goals asadditional partially observable factors in the environment. We propose a newframework Follow Instructions with Social and Embodied Reasoning FISERaiming for better natural language instruction following in collaborativeembodied tasks. Our framework makes explicit inferences about human goals andintentions as intermediate reasoning steps. We implement a set ofTransformer-based models and evaluate them over a challenging benchmarkHandMeThat. We empirically demonstrate that using social reasoning toexplicitly infer human intentions before making action plans surpasses purelyend-to-end approaches. We also compare our implementation with strongbaselines including Chain of Thought prompting on the largest availablepre-trained language models and find that FISER provides better performance onthe embodied social reasoning tasks under investigation reaching thestate-of-the-art on HandMeThat. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2409.18128v1 |
|title| FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner |
|authors| Wenliang ZhaoMinglei ShiXumin YuJie ZhouJiwen Lu
|links| http://arxiv.org/abs/2409.18128v1 |
|updated| 2024-09-26 17:59:51 UTC |
|summary| Building on the success of diffusion models in visual generation flow-basedmodels reemerge as another prominent family of generative models that haveachieved competitive or better performance in terms of both visual quality andinference speed. By learning the velocity field through flow-matchingflow-based models tend to produce a straighter sampling trajectory which isadvantageous during the sampling process. However unlike diffusion models forwhich fast samplers are well-developed efficient sampling of flow-basedgenerative models has been rarely explored. In this paper we propose aframework called FlowTurbo to accelerate the sampling of flow-based modelswhile still enhancing the sampling quality. Our primary observation is that thevelocity predictors outputs in the flow-based models will become stable duringthe sampling enabling the estimation of velocity via a lightweight velocityrefiner. Additionally we introduce several techniques including a pseudocorrector and sample-aware compilation to further reduce inference time. SinceFlowTurbo does not change the multi-step sampling paradigm it can beeffectively applied for various tasks such as image editing inpainting etc.By integrating FlowTurbo into different flow-based models we obtain anacceleration ratio of 53.1sim58.3 on class-conditional generation and29.8sim38.5 on text-to-image generation. Notably FlowTurbo reaches an FIDof 2.12 on ImageNet with 100 ms / img and FID of 3.93 with 38 ms / imgachieving the real-time image generation and establishing the newstate-of-the-art. Code is available at https://github.com/shiml20/FlowTurbo. |


| Item |Content|
| --- |---|
|idx| 2409.18127v1 |
|title| EgoLM: Multi-Modal Language Model of Egocentric Motions |
|authors| Fangzhou HongVladimir GuzovHyo Jin KimYuting YeRichard NewcombeZiwei LiuLingni Ma
|links| http://arxiv.org/abs/2409.18127v1 |
|updated| 2024-09-26 17:59:31 UTC |
|summary| As the prevalence of wearable devices learning egocentric motions becomesessential to develop contextual AI. In this work we present EgoLM a versatileframework that tracks and understands egocentric motions from multi-modalinputs e.g. egocentric videos and motion sensors. EgoLM exploits richcontexts for the disambiguation of egomotion tracking and understanding whichare ill-posed under single modality conditions. To facilitate the versatile andmulti-modal framework our key insight is to model the joint distribution ofegocentric motions and natural languages using large language models LLM.Multi-modal sensor inputs are encoded and projected to the joint latent spaceof language models and used to prompt motion generation or text generation foregomotion tracking or understanding respectively. Extensive experiments onlarge-scale multi-modal human motion dataset validate the effectiveness ofEgoLM as a generalist model for universal egocentric learning. |


| Item |Content|
| --- |---|
|idx| 2409.18125v1 |
|title| LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness |
|authors| Chenming ZhuTai WangWenwei ZhangJiangmiao PangXihui Liu
|links| http://arxiv.org/abs/2409.18125v1 |
|updated| 2024-09-26 17:59:11 UTC |
|summary| Recent advancements in Large Multimodal Models LMMs have greatly enhancedtheir proficiency in 2D visual understanding tasks enabling them toeffectively process and understand images and videos. However the developmentof LMMs with 3D-awareness for 3D scene understanding has been hindered by thelack of large-scale 3D vision-language datasets and powerful 3D encoders. Inthis paper we introduce a simple yet effective framework called LLaVA-3D.Leveraging the strong 2D understanding priors from LLaVA our LLaVA-3Defficiently adapts LLaVA for 3D scene understanding without compromising 2Dunderstanding capabilities. To achieve this we employ a simple yet effectiverepresentation 3D Patch which connects 2D CLIP patch features with theircorresponding positions in 3D space. By integrating the 3D Patches into 2D LMMsand employing joint 2D and 3D vision-language instruction tuning we establisha unified architecture for both 2D image understanding and 3D sceneunderstanding. Experimental results show that LLaVA-3D converges 3.5x fasterthan existing 3D LMMs when trained on 3D vision-language datasets. MoreoverLLaVA-3D not only achieves state-of-the-art performance across various 3D tasksbut also maintains comparable 2D image understanding and vision-languageconversation capabilities with LLaVA. |


| Item |Content|
| --- |---|
|idx| 2409.18124v1 |
|title| Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction |
|authors| Jing HeHaodong LiWei YinYixun LiangLeheng LiKaiqiang ZhouHongbo LiuBingbing LiuYing-Cong Chen
|links| http://arxiv.org/abs/2409.18124v1 |
|updated| 2024-09-26 17:58:55 UTC |
|summary| Leveraging the visual priors of pre-trained text-to-image diffusion modelsoffers a promising solution to enhance zero-shot generalization in denseprediction tasks. However existing methods often uncritically use the originaldiffusion formulation which may not be optimal due to the fundamentaldifferences between dense prediction and image generation. In this paper weprovide a systemic analysis of the diffusion formulation for the denseprediction focusing on both quality and efficiency. And we find that theoriginal parameterization type for image generation which learns to predictnoise is harmful for dense prediction the multi-step noising/denoisingdiffusion process is also unnecessary and challenging to optimize. Based onthese insights we introduce Lotus a diffusion-based visual foundation modelwith a simple yet effective adaptation protocol for dense prediction.Specifically Lotus is trained to directly predict annotations instead ofnoise thereby avoiding harmful variance. We also reformulate the diffusionprocess into a single-step procedure simplifying optimization andsignificantly boosting inference speed. Additionally we introduce a noveltuning strategy called detail preserver which achieves more accurate andfine-grained predictions. Without scaling up the training data or modelcapacity Lotus achieves SoTA performance in zero-shot depth and normalestimation across various datasets. It also significantly enhances efficiencybeing hundreds of times faster than most existing diffusion-based methods. |


| Item |Content|
| --- |---|
|idx| 2409.18121v1 |
|title| Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction |
|authors| Justin KerrChung Min KimMingxuan WuBrent YiQianqian WangKen GoldbergAngjoo Kanazawa
|links| http://arxiv.org/abs/2409.18121v1 |
|updated| 2024-09-26 17:57:16 UTC |
|summary| Humans can learn to manipulate new objects by simply watching othersproviding robots with the ability to learn from such demonstrations wouldenable a natural interface specifying new behaviors. This work develops RobotSee Robot Do RSRD a method for imitating articulated object manipulationfrom a single monocular RGB human demonstration given a single staticmulti-view object scan. We first propose 4D Differentiable Part Models4D-DPM a method for recovering 3D part motion from a monocular video withdifferentiable rendering. This analysis-by-synthesis approach uses part-centricfeature fields in an iterative optimization which enables the use of geometricregularizers to recover 3D motions from only a single video. Given this 4Dreconstruction the robot replicates object trajectories by planning bimanualarm motions that induce the demonstrated object part motion. By representingdemonstrations as part-centric trajectories RSRD focuses on replicating thedemonstrations intended behavior while considering the robots ownmorphological limits rather than attempting to reproduce the hands motion. Weevaluate 4D-DPMs 3D tracking accuracy on ground truth annotated 3D parttrajectories and RSRDs physical execution performance on 9 objects across 10trials each on a bimanual YuMi robot. Each phase of RSRD achieves an average of87 success rate for a total end-to-end success rate of 60 across 90 trials.Notably this is accomplished using only feature fields distilled from largepretrained vision models -- without any task-specific training fine-tuningdataset collection or annotation. Project page:https://robot-see-robot-do.github.io |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2409.18010v1 |
|title| End-to-end guarantees for indirect data-driven control of bilinear systems with finite stochastic data |
|authors| Nicolas ChatzikiriakosRobin SträsserFrank AllgöwerAndrea Iannelli
|links| http://arxiv.org/abs/2409.18010v1 |
|updated| 2024-09-26 16:19:49 UTC |
|summary| In this paper we propose an end-to-end algorithm for indirect data-drivencontrol for bilinear systems with stability guarantees. We consider the casewhere the collected i.i.d. data is affected by probabilistic noise withpossibly unbounded support and leverage tools from statistical learning theoryto derive finite sample identification error bounds. To this end we solve thebilinear identification problem by solving a set of linear and affineidentification problems by a particular choice of a control input during thedata collection phase. We provide a priori as well as data-dependent finitesample identification error bounds on the individual matrices as well asellipsoidal bounds both of which are structurally suitable for control.Further we integrate the structure of the derived identification error boundsin a robust controller design to obtain an exponentially stable closed-loop. Bymeans of an extensive numerical study we showcase the interplay between thecontroller design and the derived identification error bounds. Moreover wenote appealing connections of our results to indirect data-driven control ofgeneral nonlinear systems through Koopman operator theory and discuss how ourresults may be applied in this setup. |


| Item |Content|
| --- |---|
|idx| 2409.17991v1 |
|title| Dimension-independent learning rates for high-dimensional classification problems |
|authors| Andres Felipe Lerma-PinedaPhilipp PetersenSimon FriederThomas Lukasiewicz
|links| http://arxiv.org/abs/2409.17991v1 |
|updated| 2024-09-26 16:02:13 UTC |
|summary| We study the problem of approximating and estimating classification functionsthat have their decision boundary in the RBV2 space. Functions of RBV2type arise naturally as solutions of regularized neural network learningproblems and neural networks can approximate these functions without the curseof dimensionality. We modify existing results to show that every RBV2function can be approximated by a neural network with bounded weights.Thereafter we prove the existence of a neural network with bounded weightsapproximating a classification function. And we leverage these bounds toquantify the estimation rates. Finally we present a numerical study thatanalyzes the effect of different regularity conditions on the decisionboundaries. |


| Item |Content|
| --- |---|
|idx| 2409.17858v1 |
|title| How Feature Learning Can Improve Neural Scaling Laws |
|authors| Blake BordelonAlexander AtanasovCengiz Pehlevan
|links| http://arxiv.org/abs/2409.17858v1 |
|updated| 2024-09-26 14:05:32 UTC |
|summary| We develop a solvable model of neural scaling laws beyond the kernel limit.Theoretical analysis of this model shows how performance scales with modelsize training time and the total amount of available data. We identify threescaling regimes corresponding to varying task difficulties: hard easy andsuper easy tasks. For easy and super-easy target functions which lie in thereproducing kernel Hilbert space RKHS defined by the initial infinite-widthNeural Tangent Kernel NTK the scaling exponents remain unchanged betweenfeature learning and kernel regime models. For hard tasks defined as thoseoutside the RKHS of the initial NTK we demonstrate both analytically andempirically that feature learning can improve scaling with training time andcompute nearly doubling the exponent for hard tasks. This leads to a differentcompute optimal strategy to scale parameters and training time in the featurelearning regime. We support our finding that feature learning improves thescaling law for hard tasks but not for easy and super-easy tasks withexperiments of nonlinear MLPs fitting functions with power-law Fourier spectraon the circle and CNNs learning vision tasks. |


| Item |Content|
| --- |---|
|idx| 2409.17804v1 |
|title| Enriched Functional Tree-Based Classifiers: A Novel Approach Leveraging Derivatives and Geometric Features |
|authors| Fabrizio MaturoAnnamaria Porreca
|links| http://arxiv.org/abs/2409.17804v1 |
|updated| 2024-09-26 12:57:47 UTC |
|summary| The positioning of this research falls within the scalar-on-functionclassification literature a field of significant interest across variousdomains particularly in statistics mathematics and computer science. Thisstudy introduces an advanced methodology for supervised classification byintegrating Functional Data Analysis FDA with tree-based ensemble techniquesfor classifying high-dimensional time series. The proposed framework EnrichedFunctional Tree-Based Classifiers EFTCs leverages derivative and geometricfeatures benefiting from the diversity inherent in ensemble methods to furtherenhance predictive performance and reduce variance. While our approach has beentested on the enrichment of Functional Classification Trees FCTs FunctionalK-NN FKNN Functional Random Forest FRF Functional XGBoost FXGB andFunctional LightGBM FLGBM it could be extended to other tree-based andnon-tree-based classifiers with appropriate considerations emerging from thisinvestigation. Through extensive experimental evaluations on seven real-worlddatasets and six simulated scenarios this proposal demonstrates fascinatingimprovements over traditional approaches providing new insights into theapplication of FDA in complex high-dimensional learning problems. |


| Item |Content|
| --- |---|
|idx| 2409.17704v1 |
|title| Transfer Learning in $\ell_1$ Regularized Regression: Hyperparameter Selection Strategy based on Sharp Asymptotic Analysis |
|authors| Koki OkajimaTomoyuki Obuchi
|links| http://arxiv.org/abs/2409.17704v1 |
|updated| 2024-09-26 10:20:59 UTC |
|summary| Transfer learning techniques aim to leverage information from multiplerelated datasets to enhance prediction quality against a target dataset. Suchmethods have been adopted in the context of high-dimensional sparse regressionand some Lasso-based algorithms have been invented: Trans-Lasso and PretrainingLasso are such examples. These algorithms require the statistician to selecthyperparameters that control the extent and type of information transfer fromrelated datasets. However selection strategies for these hyperparameters aswell as the impact of these choices on the algorithms performance have beenlargely unexplored. To address this we conduct a thorough precise study ofthe algorithm in a high-dimensional setting via an asymptotic analysis usingthe replica method. Our approach reveals a surprisingly simple behavior of thealgorithm: Ignoring one of the two types of information transferred to thefine-tuning stage has little effect on generalization performance implyingthat efforts for hyperparameter selection can be significantly reduced. Ourtheoretical findings are also empirically supported by real-world applicationson the IMDb dataset. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2409.18060v1 |
|title| Infering Alt-text For UI Icons With Large Language Models During App Development |
|authors| Sabrina HaqueChristoph Csallner
|links| http://arxiv.org/abs/2409.18060v1 |
|updated| 2024-09-26 17:01:33 UTC |
|summary| Ensuring accessibility in mobile applications remains a significantchallenge particularly for visually impaired users who rely on screen readers.User interface icons are essential for navigation and interaction and oftenlack meaningful alt-text creating barriers to effective use. Traditional deeplearning approaches for generating alt-text require extensive datasets andstruggle with the diversity and imbalance of icon types. More recent VisionLanguage Models VLMs require complete UI screens which can be impracticalduring the iterative phases of app development. To address these issues weintroduce a novel method using Large Language Models LLMs to autonomouslygenerate informative alt-text for mobile UI icons with partial UI data. Byincorporating icon context that include class resource ID boundsOCR-detected text and contextual information from parent and sibling nodes wefine-tune an off-the-shelf LLM on a small dataset of approximately 1.4k iconsyielding IconDesc. In an empirical evaluation and a user study IconDescdemonstrates significant improvements in generating relevant alt-text. Thisability makes IconDesc an invaluable tool for developers aiding in the rapiditeration and enhancement of UI accessibility. |


| Item |Content|
| --- |---|
|idx| 2409.18037v1 |
|title| HARMONIC: A Framework for Explanatory Cognitive Robots |
|authors| Sanjay OrugantiSergei NirenburgMarjorie McShaneJesse EnglishMichael K. RobertsChristian Arndt
|links| http://arxiv.org/abs/2409.18037v1 |
|updated| 2024-09-26 16:42:13 UTC |
|summary| We present HARMONIC a framework for implementing cognitive robots thattransforms general-purpose robots into trusted teammates capable of complexdecision-making natural communication and human-level explanation. Theframework supports interoperability between a strategic cognitive layer forhigh-level decision-making and a tactical robot layer for low-level controland execution. We describe the core features of the framework and our initialimplementation in which HARMONIC was deployed on a simulated UGV and droneinvolved in a multi-robot search and retrieval task. |


| Item |Content|
| --- |---|
|idx| 2409.18009v1 |
|title| Control Industrial Automation System with Large Language Models |
|authors| Yuchen XiaNasser JazdiJize ZhangChaitanya ShahMichael Weyrich
|links| http://arxiv.org/abs/2409.18009v1 |
|updated| 2024-09-26 16:19:37 UTC |
|summary| Traditional industrial automation systems require specialized expertise tooperate and complex reprogramming to adapt to new processes. Large languagemodels offer the intelligence to make them more flexible and easier to use.However LLMs application in industrial settings is underexplored. This paperintroduces a framework for integrating LLMs to achieve end-to-end control ofindustrial automation systems. At the core of the framework are an agent systemdesigned for industrial tasks a structured prompting method and anevent-driven information modeling mechanism that provides real-time data forLLM inference. The framework supplies LLMs with real-time events on differentcontext semantic levels allowing them to interpret the information generateproduction plans and control operations on the automation system. It alsosupports structured dataset creation for fine-tuning on this downstreamapplication of LLMs. Our contribution includes a formal system designproof-of-concept implementation and a method for generating task-specificdatasets for LLM fine-tuning and testing. This approach enables a more adaptiveautomation system that can respond to spontaneous events while allowing easieroperation and configuration through natural language for more intuitivehuman-machine interaction. We provide demo videos and detailed data on GitHub:https://github.com/YuchenXia/LLM4IAS |


| Item |Content|
| --- |---|
|idx| 2409.17987v1 |
|title| LLM4Brain: Training a Large Language Model for Brain Video Understanding |
|authors| Ruizhe ZhengLichao Sun
|links| http://arxiv.org/abs/2409.17987v1 |
|updated| 2024-09-26 15:57:08 UTC |
|summary| Decoding visual-semantic information from brain signals such as functionalMRI fMRI across different subjects poses significant challenges includinglow signal-to-noise ratio limited data availability and cross-subjectvariability. Recent advancements in large language models LLMs showremarkable effectiveness in processing multimodal information. In this studywe introduce an LLM-based approach for reconstructing visual-semanticinformation from fMRI signals elicited by video stimuli. Specifically weemploy fine-tuning techniques on an fMRI encoder equipped with adaptors totransform brain responses into latent representations aligned with the videostimuli. Subsequently these representations are mapped to textual modality byLLM. In particular we integrate self-supervised domain adaptation methods toenhance the alignment between visual-semantic information and brain responses.Our proposed method achieves good results using various quantitative semanticmetrics while yielding similarity with ground-truth information. |


| Item |Content|
| --- |---|
|idx| 2409.17952v1 |
|title| Participatory design: A systematic review and insights for future practice |
|authors| Peter WacnikShanna DalyAditi Verma
|links| http://arxiv.org/abs/2409.17952v1 |
|updated| 2024-09-26 15:29:34 UTC |
|summary| Participatory Design -- an iterative flexible design process that uses theclose involvement of stakeholders most often end users -- is growing in useacross design disciplines. As an increasing number of practitioners turn toParticipatory Design PD it has become less rigidly defined withstakeholders engaged to varying degrees through the use of disjointedtechniques. This ambiguous understanding can be counterproductive whendiscussing PD processes. Our findings synthesize key decisions and approachesfrom design peers that can support others in engaging in PD practice. Weinvestigated how scholars report the use of Participatory Design in the fieldthrough a systematic literature review. We found that a majority of PDliterature examined specific case studies of PD 53 of 88 articles with thedesign of intangible systems representing the most common design context 61 of88 articles. Stakeholders most often participated throughout multiple stagesof a design process 65 of 88 articles recruited in a variety of ways andengaged in several of the 14 specific participatory techniques identified. Thissystematic review provides todays practitioners synthesized learnings frompast Participatory Design processes to inform and improve future use of PDattempting to remedy inequitable design by engaging directly with stakeholdersand users. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2409.18052v1 |
|title| Explaining Explaining |
|authors| Sergei NirenburgMarjorie McShaneKenneth W. GoodmanSanjay Oruganti
|links| http://arxiv.org/abs/2409.18052v1 |
|updated| 2024-09-26 16:55:44 UTC |
|summary| Explanation is key to people having confidence in high-stakes AI systems.However machine-learning-based systems - which account for almost all currentAI - cant explain because they are usually black boxes. The explainable AIXAI movement hedges this problem by redefining explanation. Thehuman-centered explainable AI HCXAI movement identifies theexplanation-oriented needs of users but cant fulfill them because of itscommitment to machine learning. In order to achieve the kinds of explanationsneeded by real people operating in critical domains we must rethink how toapproach AI. We describe a hybrid approach to developing cognitive agents thatuses a knowledge-based infrastructure supplemented by data obtained throughmachine learning when applicable. These agents will serve as assistants tohumans who will bear ultimate responsibility for the decisions and actions ofthe human-robot team. We illustrate the explanatory potential of such agentsusing the under-the-hood panels of a demonstration system in which a team ofsimulated robots collaborates on a search task assigned by a human. |


| Item |Content|
| --- |---|
|idx| 2409.18047v1 |
|title| HARMONIC: Cognitive and Control Collaboration in Human-Robotic Teams |
|authors| Sanjay OrugantiSergei NirenburgMarjorie McShaneJesse EnglishMichael K. RobertsChristian Arndt
|links| http://arxiv.org/abs/2409.18047v1 |
|updated| 2024-09-26 16:48:21 UTC |
|summary| This paper presents a novel approach to multi-robot planning andcollaboration. We demonstrate a cognitive strategy for robots in human-robotteams that incorporates metacognition natural language communication andexplainability. The system is embodied using the HARMONIC architecture thatflexibly integrates cognitive and control capabilities across the team. Weevaluate our approach through simulation experiments involving a joint searchtask by a team of heterogeneous robots a UGV and a drone and a human. Wedetail the systems handling of complex real-world scenarios effective actioncoordination between robots with different capabilities and naturalhuman-robot communication. This work demonstrates that the robots ability toreason about plans goals and attitudes and to provide explanations foractions and decisions are essential prerequisites for realistic human-robotteaming. |


| Item |Content|
| --- |---|
|idx| 2409.18037v1 |
|title| HARMONIC: A Framework for Explanatory Cognitive Robots |
|authors| Sanjay OrugantiSergei NirenburgMarjorie McShaneJesse EnglishMichael K. RobertsChristian Arndt
|links| http://arxiv.org/abs/2409.18037v1 |
|updated| 2024-09-26 16:42:13 UTC |
|summary| We present HARMONIC a framework for implementing cognitive robots thattransforms general-purpose robots into trusted teammates capable of complexdecision-making natural communication and human-level explanation. Theframework supports interoperability between a strategic cognitive layer forhigh-level decision-making and a tactical robot layer for low-level controland execution. We describe the core features of the framework and our initialimplementation in which HARMONIC was deployed on a simulated UGV and droneinvolved in a multi-robot search and retrieval task. |


| Item |Content|
| --- |---|
|idx| 2409.18009v1 |
|title| Control Industrial Automation System with Large Language Models |
|authors| Yuchen XiaNasser JazdiJize ZhangChaitanya ShahMichael Weyrich
|links| http://arxiv.org/abs/2409.18009v1 |
|updated| 2024-09-26 16:19:37 UTC |
|summary| Traditional industrial automation systems require specialized expertise tooperate and complex reprogramming to adapt to new processes. Large languagemodels offer the intelligence to make them more flexible and easier to use.However LLMs application in industrial settings is underexplored. This paperintroduces a framework for integrating LLMs to achieve end-to-end control ofindustrial automation systems. At the core of the framework are an agent systemdesigned for industrial tasks a structured prompting method and anevent-driven information modeling mechanism that provides real-time data forLLM inference. The framework supplies LLMs with real-time events on differentcontext semantic levels allowing them to interpret the information generateproduction plans and control operations on the automation system. It alsosupports structured dataset creation for fine-tuning on this downstreamapplication of LLMs. Our contribution includes a formal system designproof-of-concept implementation and a method for generating task-specificdatasets for LLM fine-tuning and testing. This approach enables a more adaptiveautomation system that can respond to spontaneous events while allowing easieroperation and configuration through natural language for more intuitivehuman-machine interaction. We provide demo videos and detailed data on GitHub:https://github.com/YuchenXia/LLM4IAS |


| Item |Content|
| --- |---|
|idx| 2409.17945v1 |
|title| Modular Autonomous Vehicle in Heterogeneous Traffic Flow: Modeling, Simulation, and Implication |
|authors| Lanhang YeToshiyuki Yamamoto
|links| http://arxiv.org/abs/2409.17945v1 |
|updated| 2024-09-26 15:20:21 UTC |
|summary| Modular autonomous vehicles MAVs represent a groundbreaking concept thatintegrates modularity into the ongoing development of autonomous vehicles. Thisinnovative design introduces unique features to traffic flow allowing multiplemodules to seamlessly join together and operate collectively. To understand thetraffic flow characteristics involving these vehicles and their collectiveoperations this study established a modeling framework specifically designedto simulate their behavior within traffic flow. The mixed traffic flowincorporating arbitrarily formed trains of various modular sizes is modeledand studied. Simulations are conducted under varying levels of traffic demandand penetration rates to examine the traffic flow dynamics in the presence ofthese vehicles and their operations. The microscopic trajectories MAV traincompositions and macroscopic fundamental diagrams of the mixed traffic floware analyzed. The simulation findings indicate that integrating MAVs and theircollective operations can substantially enhance capacity with the extent ofimprovement depending on the penetration rate in mixed traffic flow. Notablythe capacity nearly doubles when the penetration rate exceeds 75. Furthermoretheir presence significantly influences and regulates the free-flow speed ofthe mixed traffic. Particularly when variations in operational speed limitsexist between the MAVs and the background traffic the mixed traffic adjusts tothe operating velocity of these vehicles. This study provides insights intopotential future traffic flow systems incorporating emerging MAV technologies. |


