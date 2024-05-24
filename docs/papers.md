# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.14863v1 |
|title| A Nurse is Blue and Elephant is Rugby: Cross Domain Alignment in Large Language Models Reveal Human-like Patterns |
|authors| Asaf YehudaiTaelin KaridiGabriel StanovskyAriel GoldsteinOmri Abend
|links| http://arxiv.org/abs/2405.14863v1 |
|updated| 2024-05-23 17:59:26 UTC |
|summary| Cross-domain alignment refers to the task of mapping a concept from onedomain to another. For example If a textitdoctor were a textitcolorwhat color would it be. This seemingly peculiar task is designed toinvestigate how people represent concrete and abstract concepts through theirmappings between categories and their reasoning processes over those mappings.In this paper we adapt this task from cognitive science to evaluate theconceptualization and reasoning abilities of large language models LLMsthrough a behavioral study. We examine several LLMs by prompting them with across-domain mapping task and analyzing their responses at both the populationand individual levels. Additionally we assess the models ability to reasonabout their predictions by analyzing and categorizing their explanations forthese mappings. The results reveal several similarities between humans andmodels mappings and explanations suggesting that models represent conceptssimilarly to humans. This similarity is evident not only in the modelrepresentation but also in their behavior. Furthermore the models mostlyprovide valid explanations and deploy reasoning paths that are similar to thoseof humans. |


| Item |Content|
| --- |---|
|idx| 2405.14862v1 |
|title| Bitune: Bidirectional Instruction-Tuning |
|authors| Dawid J. KopiczkoTijmen BlankevoortYuki M. Asano
|links| http://arxiv.org/abs/2405.14862v1 |
|updated| 2024-05-23 17:59:22 UTC |
|summary| We introduce Bitune a method that improves instruction-tuning of pretraineddecoder-only large language models leading to consistent gains on downstreamtasks. Bitune applies both causal and bidirectional attention to the prompt toobtain a better representation of the query or instruction. We realize this byintroducing two sets of parameters for which we apply parameter-efficientfinetuning techniques. These causal and bidirectional features are thencombined into a weighted average with trainable coefficients which issubsequently used to generate new tokens. We demonstrate significantimprovements in zero-shot performance on commonsense reasoning arithmetic andlanguage understanding tasks while extensive ablation studies validate therole of each component and demonstrate the methods agnosticism to differentPEFT techniques. |


| Item |Content|
| --- |---|
|idx| 2405.14839v1 |
|title| A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis |
|authors| Yue YangMona GandhiYufei WangYifan WuMichael S. YaoChris Callison-BurchJames C. GeeMark Yatskar
|links| http://arxiv.org/abs/2405.14839v1 |
|updated| 2024-05-23 17:55:02 UTC |
|summary| While deep networks have achieved broad success in analyzing natural imageswhen applied to medical scans they often fail in unexcepted situations. Weinvestigate this challenge and focus on model sensitivity to domain shiftssuch as data sampled from different hospitals or data confounded by demographicvariables such as sex race etc in the context of chest X-rays and skinlesion images. A key finding we show empirically is that existing visualbackbones lack an appropriate prior from the architecture for reliablegeneralization in these settings. Taking inspiration from medical training wepropose giving deep networks a prior grounded in explicit medical knowledgecommunicated in natural language. To this end we introduce Knowledge-enhancedBottlenecks KnoBo a class of concept bottleneck models that incorporatesknowledge priors that constrain it to reason with clinically relevant factorsfound in medical textbooks or PubMed. KnoBo uses retrieval-augmented languagemodels to design an appropriate concept space paired with an automatic trainingprocedure for recognizing the concept. We evaluate different resources ofknowledge and recognition architectures on a broad range of domain shiftsacross 20 datasets. In our comprehensive evaluation with two imagingmodalities KnoBo outperforms fine-tuned models on confounded datasets by 32.4on average. Finally evaluations reveal that PubMed is a promising resource formaking medical models less sensitive to domain shift outperforming otherresources on both diversity of information and final prediction performance. |


| Item |Content|
| --- |---|
|idx| 2405.14838v1 |
|title| From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step |
|authors| Yuntian DengYejin ChoiStuart Shieber
|links| http://arxiv.org/abs/2405.14838v1 |
|updated| 2024-05-23 17:54:14 UTC |
|summary| When leveraging language models for reasoning tasks generating explicitchain-of-thought CoT steps often proves essential for achieving high accuracyin final outputs. In this paper we investigate if models can be taught tointernalize these CoT steps. To this end we propose a simple yet effectivemethod for internalizing CoT steps: starting with a model trained for explicitCoT reasoning we gradually remove the intermediate steps and finetune themodel. This process allows the model to internalize the intermediate reasoningsteps thus simplifying the reasoning process while maintaining highperformance. Our approach enables a GPT-2 Small model to solve 9-by-9multiplication with up to 99 accuracy whereas standard training cannot solvebeyond 4-by-4 multiplication. Furthermore our method proves effective onlarger language models such as Mistral 7B achieving over 50 accuracy onGSM8K without producing any intermediate steps. |


| Item |Content|
| --- |---|
|idx| 2405.14831v1 |
|title| HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models |
|authors| Bernal Jiménez GutiérrezYiheng ShuYu GuMichihiro YasunagaYu Su
|links| http://arxiv.org/abs/2405.14831v1 |
|updated| 2024-05-23 17:47:55 UTC |
|summary| In order to thrive in hostile and ever-changing natural environmentsmammalian brains evolved to store large amounts of knowledge about the worldand continually integrate new information while avoiding catastrophicforgetting. Despite the impressive accomplishments large language modelsLLMs even with retrieval-augmented generation RAG still struggle toefficiently and effectively integrate a large amount of new experiences afterpre-training. In this work we introduce HippoRAG a novel retrieval frameworkinspired by the hippocampal indexing theory of human long-term memory to enabledeeper and more efficient knowledge integration over new experiences. HippoRAGsynergistically orchestrates LLMs knowledge graphs and the PersonalizedPageRank algorithm to mimic the different roles of neocortex and hippocampus inhuman memory. We compare HippoRAG with existing RAG methods on multi-hopquestion answering and show that our method outperforms the state-of-the-artmethods remarkably by up to 20. Single-step retrieval with HippoRAG achievescomparable or better performance than iterative retrieval like IRCoT whilebeing 10-30 times cheaper and 6-13 times faster and integrating HippoRAG intoIRCoT brings further substantial gains. Finally we show that our method cantackle new types of scenarios that are out of reach of existing methods. Codeand data are available at https://github.com/OSU-NLP-Group/HippoRAG. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.14869v1 |
|title| PuzzleAvatar: Assembling 3D Avatars from Personal Albums |
|authors| Yuliang XiuYufei YeZhen LiuDimitrios TzionasMichael J. Black
|links| http://arxiv.org/abs/2405.14869v1 |
|updated| 2024-05-23 17:59:56 UTC |
|summary| Generating personalized 3D avatars is crucial for AR/VR. However recenttext-to-3D methods that generate avatars for celebrities or fictionalcharacters struggle with everyday people. Methods for faithful reconstructiontypically require full-body images in controlled settings. What if a user couldjust upload their personal OOTD Outfit Of The Day photo collection and geta faithful avatar in return The challenge is that such casual photocollections contain diverse poses challenging viewpoints cropped views andocclusion albeit with a consistent outfit accessories and hairstyle. Weaddress this novel Album2Human task by developing PuzzleAvatar a novel modelthat generates a faithful 3D avatar in a canonical pose from a personal OOTDalbum while bypassing the challenging estimation of body and camera pose. Tothis end we fine-tune a foundational vision-language model VLM on suchphotos encoding the appearance identity garments hairstyles andaccessories of a person into separate learned tokens and instilling thesecues into the VLM. In effect we exploit the learned tokens as puzzle piecesfrom which we assemble a faithful personalized 3D avatar. Importantly we cancustomize avatars by simply inter-changing tokens. As a benchmark for this newtask we collect a new dataset called PuzzleIOI with 41 subjects in a totalof nearly 1K OOTD configurations in challenging partial photos with pairedground-truth 3D bodies. Evaluation shows that PuzzleAvatar not only has highreconstruction accuracy outperforming TeCH and MVDreamBooth but also a uniquescalability to album photos and strong robustness. Our model and data will bepublic. |


| Item |Content|
| --- |---|
|idx| 2405.14868v1 |
|title| Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis |
|authors| Basile Van HoorickRundi WuEge OzgurogluKyle SargentRuoshi LiuPavel TokmakovAchal DaveChangxi ZhengCarl Vondrick
|links| http://arxiv.org/abs/2405.14868v1 |
|updated| 2024-05-23 17:59:52 UTC |
|summary| Accurate reconstruction of complex dynamic scenes from just a singleviewpoint continues to be a challenging task in computer vision. Currentdynamic novel view synthesis methods typically require videos from manydifferent camera viewpoints necessitating careful recording setups andsignificantly restricting their utility in the wild as well as in terms ofembodied AI applications. In this paper we propose textbfGCD acontrollable monocular dynamic view synthesis pipeline that leverageslarge-scale diffusion priors to given a video of any scene generate asynchronous video from any other chosen perspective conditioned on a set ofrelative camera pose parameters. Our model does not require depth as input anddoes not explicitly model 3D scene geometry instead performing end-to-endvideo-to-video translation in order to achieve its goal efficiently. Despitebeing trained on synthetic multi-view video data only zero-shot real-worldgeneralization experiments show promising results in multiple domainsincluding robotics object permanence and driving environments. We believe ourframework can potentially unlock powerful applications in rich dynamic sceneunderstanding perception for robotics and interactive 3D video viewingexperiences for virtual reality. |


| Item |Content|
| --- |---|
|idx| 2405.14863v1 |
|title| A Nurse is Blue and Elephant is Rugby: Cross Domain Alignment in Large Language Models Reveal Human-like Patterns |
|authors| Asaf YehudaiTaelin KaridiGabriel StanovskyAriel GoldsteinOmri Abend
|links| http://arxiv.org/abs/2405.14863v1 |
|updated| 2024-05-23 17:59:26 UTC |
|summary| Cross-domain alignment refers to the task of mapping a concept from onedomain to another. For example If a textitdoctor were a textitcolorwhat color would it be. This seemingly peculiar task is designed toinvestigate how people represent concrete and abstract concepts through theirmappings between categories and their reasoning processes over those mappings.In this paper we adapt this task from cognitive science to evaluate theconceptualization and reasoning abilities of large language models LLMsthrough a behavioral study. We examine several LLMs by prompting them with across-domain mapping task and analyzing their responses at both the populationand individual levels. Additionally we assess the models ability to reasonabout their predictions by analyzing and categorizing their explanations forthese mappings. The results reveal several similarities between humans andmodels mappings and explanations suggesting that models represent conceptssimilarly to humans. This similarity is evident not only in the modelrepresentation but also in their behavior. Furthermore the models mostlyprovide valid explanations and deploy reasoning paths that are similar to thoseof humans. |


| Item |Content|
| --- |---|
|idx| 2405.14861v1 |
|title| Adapting to Unknown Low-Dimensional Structures in Score-Based Diffusion Models |
|authors| Gen LiYuling Yan
|links| http://arxiv.org/abs/2405.14861v1 |
|updated| 2024-05-23 17:59:10 UTC |
|summary| This paper investigates score-based diffusion models when the underlyingtarget distribution is concentrated on or near low-dimensional manifolds withinthe higher-dimensional space in which they formally reside a commoncharacteristic of natural image distributions. Despite previous efforts tounderstand the data generation process of diffusion models existingtheoretical support remains highly suboptimal in the presence oflow-dimensional structure which we strengthen in this paper. For the popularDenoising Diffusion Probabilistic Model DDPM we find that the dependency ofthe error incurred within each denoising step on the ambient dimension d isin general unavoidable. We further identify a unique design of coefficientsthat yields a converges rate at the order of Ok2/sqrtT up to logfactors where k is the intrinsic dimension of the target distribution andT is the number of steps. This represents the first theoretical demonstrationthat the DDPM sampler can adapt to unknown low-dimensional structures in thetarget distribution highlighting the critical importance of coefficientdesign. All of this is achieved by a novel set of analysis tools thatcharacterize the algorithmic dynamics in a more deterministic manner. |


| Item |Content|
| --- |---|
|idx| 2405.14857v1 |
|title| Semantica: An Adaptable Image-Conditioned Diffusion Model |
|authors| Manoj KumarNeil HoulsbyEmiel Hoogeboom
|links| http://arxiv.org/abs/2405.14857v1 |
|updated| 2024-05-23 17:58:03 UTC |
|summary| We investigate the task of adapting image generative models to differentdatasets without finetuneing. To this end we introduce Semantica animage-conditioned diffusion model capable of generating images based on thesemantics of a conditioning image. Semantica is trained exclusively onweb-scale image pairs that is it receives a random image from a webpage asconditional input and models another random image from the same webpage. Ourexperiments highlight the expressivity of pretrained image encoders andnecessity of semantic-based data filtering in achieving high-quality imagegeneration. Once trained it can adaptively generate new images from a datasetby simply using images from that dataset as input. We study the transferproperties of Semantica on ImageNet LSUN Churches LSUN Bedroom and SUN397. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.14868v1 |
|title| Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis |
|authors| Basile Van HoorickRundi WuEge OzgurogluKyle SargentRuoshi LiuPavel TokmakovAchal DaveChangxi ZhengCarl Vondrick
|links| http://arxiv.org/abs/2405.14868v1 |
|updated| 2024-05-23 17:59:52 UTC |
|summary| Accurate reconstruction of complex dynamic scenes from just a singleviewpoint continues to be a challenging task in computer vision. Currentdynamic novel view synthesis methods typically require videos from manydifferent camera viewpoints necessitating careful recording setups andsignificantly restricting their utility in the wild as well as in terms ofembodied AI applications. In this paper we propose textbfGCD acontrollable monocular dynamic view synthesis pipeline that leverageslarge-scale diffusion priors to given a video of any scene generate asynchronous video from any other chosen perspective conditioned on a set ofrelative camera pose parameters. Our model does not require depth as input anddoes not explicitly model 3D scene geometry instead performing end-to-endvideo-to-video translation in order to achieve its goal efficiently. Despitebeing trained on synthetic multi-view video data only zero-shot real-worldgeneralization experiments show promising results in multiple domainsincluding robotics object permanence and driving environments. We believe ourframework can potentially unlock powerful applications in rich dynamic sceneunderstanding perception for robotics and interactive 3D video viewingexperiences for virtual reality. |


| Item |Content|
| --- |---|
|idx| 2405.14863v1 |
|title| A Nurse is Blue and Elephant is Rugby: Cross Domain Alignment in Large Language Models Reveal Human-like Patterns |
|authors| Asaf YehudaiTaelin KaridiGabriel StanovskyAriel GoldsteinOmri Abend
|links| http://arxiv.org/abs/2405.14863v1 |
|updated| 2024-05-23 17:59:26 UTC |
|summary| Cross-domain alignment refers to the task of mapping a concept from onedomain to another. For example If a textitdoctor were a textitcolorwhat color would it be. This seemingly peculiar task is designed toinvestigate how people represent concrete and abstract concepts through theirmappings between categories and their reasoning processes over those mappings.In this paper we adapt this task from cognitive science to evaluate theconceptualization and reasoning abilities of large language models LLMsthrough a behavioral study. We examine several LLMs by prompting them with across-domain mapping task and analyzing their responses at both the populationand individual levels. Additionally we assess the models ability to reasonabout their predictions by analyzing and categorizing their explanations forthese mappings. The results reveal several similarities between humans andmodels mappings and explanations suggesting that models represent conceptssimilarly to humans. This similarity is evident not only in the modelrepresentation but also in their behavior. Furthermore the models mostlyprovide valid explanations and deploy reasoning paths that are similar to thoseof humans. |


| Item |Content|
| --- |---|
|idx| 2405.14861v1 |
|title| Adapting to Unknown Low-Dimensional Structures in Score-Based Diffusion Models |
|authors| Gen LiYuling Yan
|links| http://arxiv.org/abs/2405.14861v1 |
|updated| 2024-05-23 17:59:10 UTC |
|summary| This paper investigates score-based diffusion models when the underlyingtarget distribution is concentrated on or near low-dimensional manifolds withinthe higher-dimensional space in which they formally reside a commoncharacteristic of natural image distributions. Despite previous efforts tounderstand the data generation process of diffusion models existingtheoretical support remains highly suboptimal in the presence oflow-dimensional structure which we strengthen in this paper. For the popularDenoising Diffusion Probabilistic Model DDPM we find that the dependency ofthe error incurred within each denoising step on the ambient dimension d isin general unavoidable. We further identify a unique design of coefficientsthat yields a converges rate at the order of Ok2/sqrtT up to logfactors where k is the intrinsic dimension of the target distribution andT is the number of steps. This represents the first theoretical demonstrationthat the DDPM sampler can adapt to unknown low-dimensional structures in thetarget distribution highlighting the critical importance of coefficientdesign. All of this is achieved by a novel set of analysis tools thatcharacterize the algorithmic dynamics in a more deterministic manner. |


| Item |Content|
| --- |---|
|idx| 2405.14860v1 |
|title| Not All Language Model Features Are Linear |
|authors| Joshua EngelsIsaac LiaoEric J. MichaudWes GurneeMax Tegmark
|links| http://arxiv.org/abs/2405.14860v1 |
|updated| 2024-05-23 17:59:04 UTC |
|summary| Recent work has proposed the linear representation hypothesis: that languagemodels perform computation by manipulating one-dimensional representations ofconcepts features in activation space. In contrast we explore whether somelanguage model representations may be inherently multi-dimensional. We begin bydeveloping a rigorous definition of irreducible multi-dimensional featuresbased on whether they can be decomposed into either independent ornon-co-occurring lower-dimensional features. Motivated by these definitions wedesign a scalable method that uses sparse autoencoders to automatically findmulti-dimensional features in GPT-2 and Mistral 7B. These auto-discoveredfeatures include strikingly interpretable examples e.g. circular featuresrepresenting days of the week and months of the year. We identify tasks wherethese exact circles are used to solve computational problems involving modulararithmetic in days of the week and months of the year. Finally we provideevidence that these circular features are indeed the fundamental unit ofcomputation in these tasks with intervention experiments on Mistral 7B andLlama 3 8B and we find further circular representations by breaking down thehidden states for these tasks into interpretable components. |


| Item |Content|
| --- |---|
|idx| 2405.14857v1 |
|title| Semantica: An Adaptable Image-Conditioned Diffusion Model |
|authors| Manoj KumarNeil HoulsbyEmiel Hoogeboom
|links| http://arxiv.org/abs/2405.14857v1 |
|updated| 2024-05-23 17:58:03 UTC |
|summary| We investigate the task of adapting image generative models to differentdatasets without finetuneing. To this end we introduce Semantica animage-conditioned diffusion model capable of generating images based on thesemantics of a conditioning image. Semantica is trained exclusively onweb-scale image pairs that is it receives a random image from a webpage asconditional input and models another random image from the same webpage. Ourexperiments highlight the expressivity of pretrained image encoders andnecessity of semantic-based data filtering in achieving high-quality imagegeneration. Once trained it can adaptively generate new images from a datasetby simply using images from that dataset as input. We study the transferproperties of Semantica on ImageNet LSUN Churches LSUN Bedroom and SUN397. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.14873v1 |
|title| Federated Online Adaptation for Deep Stereo |
|authors| Matteo PoggiFabio Tosi
|links| http://arxiv.org/abs/2405.14873v1 |
|updated| 2024-05-23 17:59:58 UTC |
|summary| We introduce a novel approach for adapting deep stereo networks in acollaborative manner. By building over principles of federated learning wedevelop a distributed framework allowing for demanding the optimization processto a number of clients deployed in different environments. This makes itpossible for a deep stereo network running on resourced-constrained devicesto capitalize on the adaptation process carried out by other instances of thesame architecture and thus improve its accuracy in challenging environmentseven when it cannot carry out adaptation on its own. Experimental results showhow federated adaptation performs equivalently to on-device adaptation andeven better when dealing with challenging environments. |


| Item |Content|
| --- |---|
|idx| 2405.14870v1 |
|title| An Empirical Study of Training State-of-the-Art LiDAR Segmentation Models |
|authors| Jiahao SunXiang XuLingdong KongYouquan LiuLi LiChenming ZhuJingwei ZhangZeqi XiaoRunnan ChenTai WangWenwei ZhangKai ChenChunmei Qing
|links| http://arxiv.org/abs/2405.14870v1 |
|updated| 2024-05-23 17:59:57 UTC |
|summary| In the rapidly evolving field of autonomous driving precise segmentation ofLiDAR data is crucial for understanding complex 3D environments. Traditionalapproaches often rely on disparate standalone codebases hindering unifiedadvancements and fair benchmarking across models. To address these challengeswe introduce MMDetection3D-lidarseg a comprehensive toolbox designed for theefficient training and evaluation of state-of-the-art LiDAR segmentationmodels. We support a wide range of segmentation models and integrate advanceddata augmentation techniques to enhance robustness and generalization.Additionally the toolbox provides support for multiple leading sparseconvolution backends optimizing computational efficiency and performance. Byfostering a unified framework MMDetection3D-lidarseg streamlines developmentand benchmarking setting new standards for research and application. Ourextensive benchmark experiments on widely-used datasets demonstrate theeffectiveness of the toolbox. The codebase and trained models have beenpublicly available promoting further research and innovation in the field ofLiDAR segmentation for autonomous driving. |


| Item |Content|
| --- |---|
|idx| 2405.14871v1 |
|title| NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections |
|authors| Dor VerbinPratul P. SrinivasanPeter HedmanBen MildenhallBenjamin AttalRichard SzeliskiJonathan T. Barron
|links| http://arxiv.org/abs/2405.14871v1 |
|updated| 2024-05-23 17:59:57 UTC |
|summary| Neural Radiance Fields NeRFs typically struggle to reconstruct and renderhighly specular objects whose appearance varies quickly with changes inviewpoint. Recent works have improved NeRFs ability to render detailedspecular appearance of distant environment illumination but are unable tosynthesize consistent reflections of closer content. Moreover these techniquesrely on large computationally-expensive neural networks to model outgoingradiance which severely limits optimization and rendering speed. We addressthese issues with an approach based on ray tracing: instead of querying anexpensive neural network for the outgoing view-dependent radiance at pointsalong each camera ray our model casts reflection rays from these points andtraces them through the NeRF representation to render feature vectors which aredecoded into color using a small inexpensive network. We demonstrate that ourmodel outperforms prior methods for view synthesis of scenes containing shinyobjects and that it is the only existing NeRF method that can synthesizephotorealistic specular appearance and reflections in real-world scenes whilerequiring comparable optimization time to current state-of-the-art viewsynthesis models. |


| Item |Content|
| --- |---|
|idx| 2405.14869v1 |
|title| PuzzleAvatar: Assembling 3D Avatars from Personal Albums |
|authors| Yuliang XiuYufei YeZhen LiuDimitrios TzionasMichael J. Black
|links| http://arxiv.org/abs/2405.14869v1 |
|updated| 2024-05-23 17:59:56 UTC |
|summary| Generating personalized 3D avatars is crucial for AR/VR. However recenttext-to-3D methods that generate avatars for celebrities or fictionalcharacters struggle with everyday people. Methods for faithful reconstructiontypically require full-body images in controlled settings. What if a user couldjust upload their personal OOTD Outfit Of The Day photo collection and geta faithful avatar in return The challenge is that such casual photocollections contain diverse poses challenging viewpoints cropped views andocclusion albeit with a consistent outfit accessories and hairstyle. Weaddress this novel Album2Human task by developing PuzzleAvatar a novel modelthat generates a faithful 3D avatar in a canonical pose from a personal OOTDalbum while bypassing the challenging estimation of body and camera pose. Tothis end we fine-tune a foundational vision-language model VLM on suchphotos encoding the appearance identity garments hairstyles andaccessories of a person into separate learned tokens and instilling thesecues into the VLM. In effect we exploit the learned tokens as puzzle piecesfrom which we assemble a faithful personalized 3D avatar. Importantly we cancustomize avatars by simply inter-changing tokens. As a benchmark for this newtask we collect a new dataset called PuzzleIOI with 41 subjects in a totalof nearly 1K OOTD configurations in challenging partial photos with pairedground-truth 3D bodies. Evaluation shows that PuzzleAvatar not only has highreconstruction accuracy outperforming TeCH and MVDreamBooth but also a uniquescalability to album photos and strong robustness. Our model and data will bepublic. |


| Item |Content|
| --- |---|
|idx| 2405.14868v1 |
|title| Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis |
|authors| Basile Van HoorickRundi WuEge OzgurogluKyle SargentRuoshi LiuPavel TokmakovAchal DaveChangxi ZhengCarl Vondrick
|links| http://arxiv.org/abs/2405.14868v1 |
|updated| 2024-05-23 17:59:52 UTC |
|summary| Accurate reconstruction of complex dynamic scenes from just a singleviewpoint continues to be a challenging task in computer vision. Currentdynamic novel view synthesis methods typically require videos from manydifferent camera viewpoints necessitating careful recording setups andsignificantly restricting their utility in the wild as well as in terms ofembodied AI applications. In this paper we propose textbfGCD acontrollable monocular dynamic view synthesis pipeline that leverageslarge-scale diffusion priors to given a video of any scene generate asynchronous video from any other chosen perspective conditioned on a set ofrelative camera pose parameters. Our model does not require depth as input anddoes not explicitly model 3D scene geometry instead performing end-to-endvideo-to-video translation in order to achieve its goal efficiently. Despitebeing trained on synthetic multi-view video data only zero-shot real-worldgeneralization experiments show promising results in multiple domainsincluding robotics object permanence and driving environments. We believe ourframework can potentially unlock powerful applications in rich dynamic sceneunderstanding perception for robotics and interactive 3D video viewingexperiences for virtual reality. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.14861v1 |
|title| Adapting to Unknown Low-Dimensional Structures in Score-Based Diffusion Models |
|authors| Gen LiYuling Yan
|links| http://arxiv.org/abs/2405.14861v1 |
|updated| 2024-05-23 17:59:10 UTC |
|summary| This paper investigates score-based diffusion models when the underlyingtarget distribution is concentrated on or near low-dimensional manifolds withinthe higher-dimensional space in which they formally reside a commoncharacteristic of natural image distributions. Despite previous efforts tounderstand the data generation process of diffusion models existingtheoretical support remains highly suboptimal in the presence oflow-dimensional structure which we strengthen in this paper. For the popularDenoising Diffusion Probabilistic Model DDPM we find that the dependency ofthe error incurred within each denoising step on the ambient dimension d isin general unavoidable. We further identify a unique design of coefficientsthat yields a converges rate at the order of Ok2/sqrtT up to logfactors where k is the intrinsic dimension of the target distribution andT is the number of steps. This represents the first theoretical demonstrationthat the DDPM sampler can adapt to unknown low-dimensional structures in thetarget distribution highlighting the critical importance of coefficientdesign. All of this is achieved by a novel set of analysis tools thatcharacterize the algorithmic dynamics in a more deterministic manner. |


| Item |Content|
| --- |---|
|idx| 2405.14848v1 |
|title| Local Causal Discovery for Structural Evidence of Direct Discrimination |
|authors| Jacqueline MaaschKyra GanViolet ChenAgni OrfanoudakiNil-Jana AkpinarFei Wang
|links| http://arxiv.org/abs/2405.14848v1 |
|updated| 2024-05-23 17:56:38 UTC |
|summary| Fairness is a critical objective in policy design and algorithmicdecision-making. Identifying the causal pathways of unfairness requiresknowledge of the underlying structural causal model which may be incomplete orunavailable. This limits the practicality of causal fairness analysis incomplex or low-knowledge domains. To mitigate this practicality gap weadvocate for developing efficient causal discovery methods for fairnessapplications. To this end we introduce local discovery for directdiscrimination LD3: a polynomial-time algorithm that recovers structuralevidence of direct discrimination. LD3 performs a linear number of conditionalindependence tests with respect to variable set size. Moreover we propose agraphical criterion for identifying the weighted controlled direct effectCDE a qualitative measure of direct discrimination. We prove that thiscriterion is satisfied by the knowledge returned by LD3 increasing theaccessibility of the weighted CDE as a causal fairness measure. Taking livertransplant allocation as a case study we highlight the potential impact of LD3for modeling fairness in complex decision systems. Results on real-world datademonstrate more plausible causal relations than baselines which took 197x to5870x longer to execute. |


| Item |Content|
| --- |---|
|idx| 2405.14840v1 |
|title| Differentiable Annealed Importance Sampling Minimizes The Jensen-Shannon Divergence Between Initial and Target Distribution |
|authors| Johannes ZennRobert Bamler
|links| http://arxiv.org/abs/2405.14840v1 |
|updated| 2024-05-23 17:55:09 UTC |
|summary| Differentiable annealed importance sampling DAIS proposed by Geffner Domke 2021 and Zhang et al. 2021 allows optimizing among others over theinitial distribution of AIS. In this paper we show that in the limit of manytransitions DAIS minimizes the symmetrized KL divergence Jensen-Shannondivergence between the initial and target distribution. Thus DAIS can be seenas a form of variational inference VI in that its initial distribution is aparametric fit to an intractable target distribution. We empirically evaluatethe usefulness of the initial distribution as a variational distribution onsynthetic and real-world data observing that it often provides more accurateuncertainty estimates than standard VI optimizing the reverse KL divergenceimportance weighted VI and Markovian score climbing optimizing the forward KLdivergence. |


| Item |Content|
| --- |---|
|idx| 2405.14822v1 |
|title| PaGoDA: Progressive Growing of a One-Step Generator from a Low-Resolution Diffusion Teacher |
|authors| Dongjun KimChieh-Hsin LaiWei-Hsiang LiaoYuhta TakidaNaoki MurataToshimitsu UesakaYuki MitsufujiStefano Ermon
|links| http://arxiv.org/abs/2405.14822v1 |
|updated| 2024-05-23 17:39:09 UTC |
|summary| To accelerate sampling diffusion models DMs are often distilled intogenerators that directly map noise to data in a single step. In this approachthe resolution of the generator is fundamentally limited by that of the teacherDM. To overcome this limitation we propose Progressive Growing of DiffusionAutoencoder PaGoDA a technique to progressively grow the resolution of thegenerator beyond that of the original teacher DM. Our key insight is that apre-trained low-resolution DM can be used to deterministically encodehigh-resolution data to a structured latent space by solving the PF-ODE forwardin time data-to-noise starting from an appropriately down-sampled image.Using this frozen encoder in an auto-encoder framework we train a decoder byprogressively growing its resolution. From the nature of progressively growingdecoder PaGoDA avoids re-training teacher/student models when we upsample thestudent model making the whole training pipeline much cheaper. In experimentswe used our progressively growing decoder to upsample from the pre-trainedmodels 64x64 resolution to generate 512x512 samples achieving 2x fasterinference compared to single-step distilled Stable Diffusion like LCM. PaGoDAalso achieved state-of-the-art FIDs on ImageNet across all resolutions from64x64 to 512x512. Additionally we demonstrated PaGoDAs effectiveness insolving inverse problems and enabling controllable generation. |


| Item |Content|
| --- |---|
|idx| 2405.14806v1 |
|title| Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics |
|authors| Jonas SpinnerVictor BresóPim de HaanTilman PlehnJesse ThalerJohann Brehmer
|links| http://arxiv.org/abs/2405.14806v1 |
|updated| 2024-05-23 17:15:41 UTC |
|summary| Extracting scientific understanding from particle-physics experimentsrequires solving diverse learning problems with high precision and good dataefficiency. We propose the Lorentz Geometric Algebra Transformer L-GATr anew multi-purpose architecture for high-energy physics. L-GATr representshigh-energy data in a geometric algebra over four-dimensional space-time and isequivariant under Lorentz transformations the symmetry group of relativistickinematics. At the same time the architecture is a Transformer which makes itversatile and scalable to large systems. L-GATr is first demonstrated onregression and classification tasks from particle physics. We then constructthe first Lorentz-equivariant generative model: a continuous normalizing flowbased on an L-GATr network trained with Riemannian flow matching. Across ourexperiments L-GATr is on par with or outperforms strong domain-specificbaselines. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.14808v1 |
|title| Implicit Personalization in Language Models: A Systematic Study |
|authors| Zhijing JinNils HeilJiarui LiuShehzaad DhuliawalaYahang QiBernhard SchölkopfRada MihalceaMrinmaya Sachan
|links| http://arxiv.org/abs/2405.14808v1 |
|updated| 2024-05-23 17:18:46 UTC |
|summary| Implicit Personalization IP is a phenomenon of language models inferring ausers background from the implicit cues in the input prompts and tailoring theresponse based on this inference. While previous work has touched upon variousinstances of this problem there lacks a unified framework to study thisbehavior. This work systematically studies IP through a rigorous mathematicalformulation a multi-perspective moral reasoning framework and a set of casestudies. Our theoretical foundation for IP relies on a structural causal modeland introduces a novel method indirect intervention to estimate the causaleffect of a mediator variable that cannot be directly intervened upon. Beyondthe technical approach we also introduce a set of moral reasoning principlesbased on three schools of moral philosophy to study when IP may or may not beethically appropriate. Equipped with both mathematical and ethical insights wepresent three diverse case studies illustrating the varied nature of the IPproblem and offer recommendations for future research. Our code and data are athttps://github.com/jiarui-liu/IP. |


| Item |Content|
| --- |---|
|idx| 2405.14794v1 |
|title| RetAssist: Facilitating Vocabulary Learners with Generative Images in Story Retelling Practices |
|authors| Qiaoyi ChenSiyu LiuKaihui HuangXingbo WangXiaojuan MaJunkai ZhuZhenhui Peng
|links| http://dx.doi.org/10.1145/3643834.3661581 |
|updated| 2024-05-23 17:05:13 UTC |
|summary| Reading and repeatedly retelling a short story is a common and effectiveapproach to learning the meanings and usages of target words. However learnersoften struggle with comprehending recalling and retelling the story contextsof these target words. Inspired by the Cognitive Theory of Multimedia Learningwe propose a computational workflow to generate relevant images paired withstories. Based on the workflow we work with learners and teachers toiteratively design an interactive vocabulary learning system named RetAssist.It can generate sentence-level images of a story to facilitate theunderstanding and recall of the target words in the story retelling practices.Our within-subjects study N24 shows that compared to a baseline systemwithout generative images RetAssist significantly improves learners fluencyin expressing with target words. Participants also feel that RetAssist easestheir learning workload and is more useful. We discuss insights into leveragingtext-to-image generative models to support learning tasks. |


| Item |Content|
| --- |---|
|idx| 2405.14783v1 |
|title| Low-Energy Line Codes for On-Chip Networks |
|authors| Beyza DabakMajor GlennJingyang LiuAlexander BuckSiyi YangRobert CalderbankNatalie Enright JergerDaniel J. Sorin
|links| http://arxiv.org/abs/2405.14783v1 |
|updated| 2024-05-23 16:52:14 UTC |
|summary| Energy is a primary constraint in processor design and much of that energyis consumed in on-chip communication. Communication can be intra-core e.g.from a register file to an ALU or inter-core e.g. over the on-chip network.In this paper we use the on-chip network OCN as a case study for savingon-chip communication energy. We have identified a new way to reduce the OCNslink energy consumption by using line coding a longstanding technique ininformation theory. Our line codes called Low-Energy Line Codes LELCsreduce energy by reducing the frequency of voltage transitions of the linksand they achieve a range of energy/performance trade-offs. |


| Item |Content|
| --- |---|
|idx| 2405.14753v1 |
|title| A Transformer-Based Approach for Smart Invocation of Automatic Code Completion |
|authors| Aral de MoorArie van DeursenMaliheh Izadi
|links| http://dx.doi.org/10.1145/3664646.3664760 |
|updated| 2024-05-23 16:19:32 UTC |
|summary| Transformer-based language models are highly effective for code completionwith much research dedicated to enhancing the content of these completions.Despite their effectiveness these models come with high operational costs andcan be intrusive especially when they suggest too often and interruptdevelopers who are concentrating on their work. Current research largelyoverlooks how these models interact with developers in practice and neglects toaddress when a developer should receive completion suggestions. To tackle thisissue we developed a machine learning model that can accurately predict whento invoke a code completion tool given the code context and available telemetrydata.  To do so we collect a dataset of 200k developer interactions with ourcross-IDE code completion plugin and train several invocation filtering models.Our results indicate that our small-scale transformer model significantlyoutperforms the baseline while maintaining low enough latency. We furtherexplore the search space for integrating additional telemetry data into apre-trained transformer directly and obtain promising results. To furtherdemonstrate our approachs practical potential we deployed the model in anonline environment with 34 developers and provided real-world insights based on74k actual invocations. |


| Item |Content|
| --- |---|
|idx| 2405.14716v1 |
|title| HTN-Based Tutors: A New Intelligent Tutoring Framework Based on Hierarchical Task Networks |
|authors| Momin N. SiddiquiAdit GuptaJennifer M. ReddigChristopher J. Maclellan
|links| http://dx.doi.org/10.1145/3657604 |
|updated| 2024-05-23 15:46:42 UTC |
|summary| Intelligent tutors have shown success in delivering a personalized andadaptive learning experience. However there exist challenges regarding thegranularity of knowledge in existing frameworks and the resulting instructionsthey can provide. To address these issues we propose HTN-based tutors a newintelligent tutoring framework that represents expert models using HierarchicalTask Networks HTNs. Like other tutoring frameworks it allows flexibleencoding of different problem-solving strategies while providing the additionalbenefit of a hierarchical knowledge organization. We leverage the latter tocreate tutors that can adapt the granularity of their scaffolding. Thisorganization also aligns well with the compositional nature of skills. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.14691v1 |
|title| CityGPT: Towards Urban IoT Learning, Analysis and Interaction with Multi-Agent System |
|authors| Qinghua GuanJinhui OuyangDi WuWeiren Yu
|links| http://arxiv.org/abs/2405.14691v1 |
|updated| 2024-05-23 15:27:18 UTC |
|summary| The spatiotemporal data generated by massive sensors in the Internet ofThings IoT is extremely dynamic heterogeneous large scale andtime-dependent. It poses great challenges e.g. accuracy reliability andstability in real-time analysis and decision making for different IoTapplications. The complexity of IoT data prevents the common people fromgaining a deeper understanding of it. Agentized systems help address the lackof data insight for the common people. We propose a generic framework namelyCityGPT to facilitate the learning and analysis of IoT time series with anend-to-end paradigm. CityGPT employs three agents to accomplish thespatiotemporal analysis of IoT data. The requirement agent facilitates userinputs based on natural language. Then the analysis tasks are decomposed intotemporal and spatial analysis processes completed by corresponding dataanalysis agents temporal and spatial agents. Finally the spatiotemporalfusion agent visualizes the systems analysis results by receiving analysisresults from data analysis agents and invoking sub-visualization agents andcan provide corresponding textual descriptions based on user demands. Toincrease the insight for common people using our framework we have agnentizedthe framework facilitated by a large language model LLM to increase thedata comprehensibility. Our evaluation results on real-world data withdifferent time dependencies show that the CityGPT framework can guaranteerobust performance in IoT computing. |


| Item |Content|
| --- |---|
|idx| 2405.14546v1 |
|title| Global Behavior of Learning Dynamics in Zero-Sum Games with Memory Asymmetry |
|authors| Yuma FujimotoKaito AriuKenshi Abe
|links| http://arxiv.org/abs/2405.14546v1 |
|updated| 2024-05-23 13:25:39 UTC |
|summary| This study examines the global behavior of dynamics in learning in gamesbetween two players X and Y. We consider the simplest situation for memoryasymmetry between two players: X memorizes the other Ys previous action anduses reactive strategies while Y has no memory. Although this memorycomplicates the learning dynamics we discover two novel quantities thatcharacterize the global behavior of such complex dynamics. One is an extendedKullback-Leibler divergence from the Nash equilibrium a well-known conservedquantity from previous studies. The other is a family of Lyapunov functions ofXs reactive strategy. These two quantities capture the global behavior inwhich Xs strategy becomes more exploitative and the exploited Ys strategyconverges to the Nash equilibrium. Indeed we theoretically prove that Ysstrategy globally converges to the Nash equilibrium in the simplest gameequipped with an equilibrium in the interior of strategy spaces. Furthermoreour experiments also suggest that this global convergence is universal for moreadvanced zero-sum games than the simplest game. This study provides a novelcharacterization of the global behavior of learning in games through a coupleof indicators. |


| Item |Content|
| --- |---|
|idx| 2405.14358v1 |
|title| AI-Olympics: Exploring the Generalization of Agents through Open Competitions |
|authors| Chen WangYan SongShuai WuSa WuRuizhi ZhangShu LinHaifeng Zhang
|links| http://arxiv.org/abs/2405.14358v1 |
|updated| 2024-05-23 09:33:57 UTC |
|summary| Between 2021 and 2023 AI-Olympics a series of online AI competitions washosted by the online evaluation platform Jidi in collaboration with the IJCAIcommittee. In these competitions an agent is required to accomplish diversesports tasks in a two-dimensional continuous world while competing against anopponent. This paper provides a brief overview of the competition series andhighlights notable findings. We aim to contribute insights to the field ofmulti-agent decision-making and explore the generalization of agents throughengineering efforts. |


| Item |Content|
| --- |---|
|idx| 2405.14314v1 |
|title| Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration |
|authors| Yang ZhangShixin YangChenjia BaiFei WuXiu LiXuelong LiZhen Wang
|links| http://arxiv.org/abs/2405.14314v1 |
|updated| 2024-05-23 08:33:19 UTC |
|summary| Grounding the reasoning ability of large language models LLMs for embodiedtasks is challenging due to the complexity of the physical world. EspeciallyLLM planning for multi-agent collaboration requires communication of agents orcredit assignment as the feedback to re-adjust the proposed plans and achieveeffective coordination. However existing methods that overly rely on physicalverification or self-reflection suffer from excessive and inefficient queryingof LLMs. In this paper we propose a novel framework for multi-agentcollaboration that introduces Reinforced Advantage feedback ReAd forefficient self-refinement of plans. Specifically we perform critic regressionto learn a sequential advantage function from LLM-planned data and then treatthe LLM planner as an optimizer to generate actions that maximize the advantagefunction. It endows the LLM with the foresight to discern whether the actioncontributes to accomplishing the final task. We provide theoretical analysis byextending advantage-weighted regression in reinforcement learning tomulti-agent systems. Experiments on Overcooked-AI and a difficult variant ofRoCoBench show that ReAd surpasses baselines in success rate and alsosignificantly decreases the interaction steps of agents and query rounds ofLLMs demonstrating its high efficiency for grounding LLMs. More results aregiven at urlhttps://read-llm.github.io/. |


| Item |Content|
| --- |---|
|idx| 2405.14205v1 |
|title| Agent Planning with World Knowledge Model |
|authors| Shuofei QiaoRunnan FangNingyu ZhangYuqi ZhuXiang ChenShumin DengYong JiangPengjun XieFei HuangHuajun Chen
|links| http://arxiv.org/abs/2405.14205v1 |
|updated| 2024-05-23 06:03:19 UTC |
|summary| Recent endeavors towards directly using large language models LLMs as agentmodels to execute interactive planning tasks have shown commendable results.Despite their achievements however they still struggle with brainlesstrial-and-error in global planning and generating hallucinatory actions inlocal planning due to their poor understanding of the real physical world.Imitating humans mental world knowledge model which provides global priorknowledge before the task and maintains local dynamic knowledge during thetask in this paper we introduce parametric World Knowledge Model WKM tofacilitate agent planning. Concretely we steer the agent model toself-synthesize knowledge from both expert and sampled trajectories. Then wedevelop WKM providing prior task knowledge to guide the global planning anddynamic state knowledge to assist the local planning. Experimental results onthree complex real-world simulated datasets with three state-of-the-artopen-source LLMs Mistral-7B Gemma-7B and Llama-3-8B demonstrate that ourmethod can achieve superior performance compared to various strong baselines.Besides we analyze to illustrate that our WKM can effectively alleviate theblind trial-and-error and hallucinatory action issues providing strong supportfor the agents understanding of the world. Other interesting findings include:1 our instance-level task knowledge can generalize better to unseen tasks 2weak WKM can guide strong agent model planning and 3 unified WKM training haspromising potential for further development. Code will be available athttps://github.com/zjunlp/WKM. |


