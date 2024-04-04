# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.02904v1 |
|title| ALOHa: A New Measure for Hallucination in Captioning Models |
|authors| Suzanne PetrykDavid M. ChanAnish KachinthayaHaodi ZouJohn CannyJoseph E. GonzalezTrevor Darrell
|links| http://arxiv.org/abs/2404.02904v1 |
|updated| 2024-04-03 17:59:36 UTC |
|summary| Despite recent advances in multimodal pre-training for visual descriptionstate-of-the-art models still produce captions containing errors such ashallucinating objects not present in a scene. The existing prominent metric forobject hallucination CHAIR is limited to a fixed set of MS COCO objects andsynonyms. In this work we propose a modernized open-vocabulary metric ALOHawhich leverages large language models LLMs to measure object hallucinations.Specifically we use an LLM to extract groundable objects from a candidatecaption measure their semantic similarity to reference objects from captionsand object detections and use Hungarian matching to produce a finalhallucination score. We show that ALOHa correctly identifies 13.6 morehallucinated objects than CHAIR on HAT a new gold-standard subset of MS COCOCaptions annotated for hallucinations and 30.8 more on nocaps where objectsextend beyond MS COCO categories. Our code is available athttps://davidmchan.github.io/aloha/. |


| Item |Content|
| --- |---|
|idx| 2404.02893v1 |
|title| ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline |
|authors| Yifan XuXiao LiuXinghan LiuZhenyu HouYueyan LiXiaohan ZhangZihan WangAohan ZengZhengxiao DuWenyi ZhaoJie TangYuxiao Dong
|links| http://arxiv.org/abs/2404.02893v1 |
|updated| 2024-04-03 17:51:18 UTC |
|summary| Large language models LLMs have shown excellent mastering of humanlanguage but still struggle in real-world applications that requiremathematical problem-solving. While many strategies and datasets to enhanceLLMs mathematics are developed it remains a challenge to simultaneouslymaintain and improve both language and mathematical capabilities in deployedLLM systems.In this work we tailor the Self-Critique pipeline which addressesthe challenge in the feedback learning stage of LLM alignment. We first train ageneral Math-Critique model from the LLM itself to provide feedback signals.Then we sequentially employ rejective fine-tuning and direct preferenceoptimization over the LLMs own generations for data collection. Based onChatGLM3-32B we conduct a series of experiments on both academic and our newlycreated challenging dataset MathUserEval. Results show that our pipelinesignificantly enhances the LLMs mathematical problem-solving while stillimproving its language ability outperforming LLMs that could be two timeslarger. Related techniques have been deployed toChatGLMfootnoteurlhttps://chatglm.cn an online serving LLM. Relatedevaluation dataset and scripts are released aturlhttps://github.com/THUDM/ChatGLM-Math. |


| Item |Content|
| --- |---|
|idx| 2404.02882v1 |
|title| Linear Attention Sequence Parallelism |
|authors| Weigao SunZhen QinDong LiXuyang ShenYu QiaoYiran Zhong
|links| http://arxiv.org/abs/2404.02882v1 |
|updated| 2024-04-03 17:33:21 UTC |
|summary| Sequence Parallel SP serves as a prevalent strategy to handle longsequences that exceed the memory limit of a single GPU. However existing SPmethods do not take advantage of linear attention features resulting insub-optimal parallelism efficiency and usability for linear attention-basedlanguage models. In this paper we introduce Linear Attention Sequence ParallelLASP an efficient SP method tailored to linear attention-based languagemodels. Specifically we design an efficient point-to-point communicationmechanism to leverage the right-product kernel trick of linear attention whichsharply decreases the communication overhead of SP. We also enhance thepractical efficiency of LASP by performing kernel fusion and intermediate statecaching making the implementation of LASP hardware-friendly on GPU clusters.Furthermore we meticulously ensure the compatibility of sequence-level LASPwith all types of batch-level data parallel methods which is vital fordistributed training on large clusters with long sequences and large batches.We conduct extensive experiments on two linear attention-based models withvarying sequence lengths and GPU cluster sizes. LASP scales sequence length upto 4096K using 128 A100 80G GPUs on 1B models which is 8 times longer thanexisting SP methods while being significantly faster. The code is available athttps://github.com/OpenNLPLab/LASP. |


| Item |Content|
| --- |---|
|idx| 2404.02837v1 |
|title| Cherry on Top: Parameter Heterogeneity and Quantization in Large Language Models |
|authors| Wanyun CuiQianle Wang
|links| http://arxiv.org/abs/2404.02837v1 |
|updated| 2024-04-03 16:16:31 UTC |
|summary| This paper reveals the phenomenon of parameter heterogeneity in largelanguage models LLMs. We find that a small subset of cherry parametersexhibit a disproportionately large influence on model performance while thevast majority of parameters have minimal impact. This heterogeneity is found tobe prevalent across different model families scales and types. Motivated bythis observation we propose CherryQ a novel quantization method that unifiesthe optimization of mixed-precision parameters. CherryQ identifies andpreserves the critical cherry parameters in high precision while aggressivelyquantizing the remaining parameters to low precision. Extensive experimentsdemonstrate the effectiveness of CherryQ. CherryQ outperforms existingquantization approaches in terms of perplexity and downstream task performance.Notably our 3-bit quantized Vicuna-1.5 exhibits competitive performancecompared to their 16-bit counterparts. These findings highlight the potentialof CherryQ for enabling efficient deployment of LLMs by taking advantage ofparameter heterogeneity. |


| Item |Content|
| --- |---|
|idx| 2404.02835v1 |
|title| Retrieving Examples from Memory for Retrieval Augmented Neural Machine Translation: A Systematic Comparison |
|authors| Maxime BouthorsJosep CregoFrancois Yvon
|links| http://arxiv.org/abs/2404.02835v1 |
|updated| 2024-04-03 16:13:29 UTC |
|summary| Retrieval-Augmented Neural Machine Translation RAMT architectures retrieveexamples from memory to guide the generation process. While most works in thistrend explore new ways to exploit the retrieved examples the upstreamretrieval step is mostly unexplored. In this paper we study the effect ofvarying retrieval methods for several translation architectures to betterunderstand the interplay between these two processes. We conduct experiments intwo language pairs in a multi-domain setting and consider several downstreamarchitectures based on a standard autoregressive model an edit-based modeland a large language model with in-context learning. Our experiments show thatthe choice of the retrieval technique impacts the translation scores withvariance across architectures. We also discuss the effects of increasing thenumber and diversity of examples which are mostly positive across the board. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.02905v1 |
|title| Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction |
|authors| Keyu TianYi JiangZehuan YuanBingyue PengLiwei Wang
|links| http://arxiv.org/abs/2404.02905v1 |
|updated| 2024-04-03 17:59:53 UTC |
|summary| We present Visual AutoRegressive modeling VAR a new generation paradigmthat redefines the autoregressive learning on images as coarse-to-finenext-scale prediction or next-resolution prediction diverging from thestandard raster-scan next-token prediction. This simple intuitivemethodology allows autoregressive AR transformers to learn visualdistributions fast and generalize well: VAR for the first time makes ARmodels surpass diffusion transformers in image generation. On ImageNet 256x256benchmark VAR significantly improve AR baseline by improving Frechet inceptiondistance FID from 18.65 to 1.80 inception score IS from 80.4 to 356.4with around 20x faster inference speed. It is also empirically verified thatVAR outperforms the Diffusion Transformer DiT in multiple dimensionsincluding image quality inference speed data efficiency and scalability.Scaling up VAR models exhibits clear power-law scaling laws similar to thoseobserved in LLMs with linear correlation coefficients near -0.998 as solidevidence. VAR further showcases zero-shot generalization ability in downstreamtasks including image in-painting out-painting and editing. These resultssuggest VAR has initially emulated the two important properties of LLMs:Scaling Laws and zero-shot task generalization. We have released all models andcodes to promote the exploration of AR/VAR models for visual generation andunified learning. |


| Item |Content|
| --- |---|
|idx| 2404.02904v1 |
|title| ALOHa: A New Measure for Hallucination in Captioning Models |
|authors| Suzanne PetrykDavid M. ChanAnish KachinthayaHaodi ZouJohn CannyJoseph E. GonzalezTrevor Darrell
|links| http://arxiv.org/abs/2404.02904v1 |
|updated| 2024-04-03 17:59:36 UTC |
|summary| Despite recent advances in multimodal pre-training for visual descriptionstate-of-the-art models still produce captions containing errors such ashallucinating objects not present in a scene. The existing prominent metric forobject hallucination CHAIR is limited to a fixed set of MS COCO objects andsynonyms. In this work we propose a modernized open-vocabulary metric ALOHawhich leverages large language models LLMs to measure object hallucinations.Specifically we use an LLM to extract groundable objects from a candidatecaption measure their semantic similarity to reference objects from captionsand object detections and use Hungarian matching to produce a finalhallucination score. We show that ALOHa correctly identifies 13.6 morehallucinated objects than CHAIR on HAT a new gold-standard subset of MS COCOCaptions annotated for hallucinations and 30.8 more on nocaps where objectsextend beyond MS COCO categories. Our code is available athttps://davidmchan.github.io/aloha/. |


| Item |Content|
| --- |---|
|idx| 2404.02900v1 |
|title| DeiT-LT Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets |
|authors| Harsh RangwaniPradipto MondalMayank MishraAshish Ramayee AsokanR. Venkatesh Babu
|links| http://arxiv.org/abs/2404.02900v1 |
|updated| 2024-04-03 17:58:21 UTC |
|summary| Vision Transformer ViT has emerged as a prominent architecture for variouscomputer vision tasks. In ViT we divide the input image into patch tokens andprocess them through a stack of self attention blocks. However unlikeConvolutional Neural Networks CNN ViTs simple architecture has noinformative inductive bias e.g. localityetc. . Due to this ViT requires alarge amount of data for pre-training. Various data efficient approaches DeiThave been proposed to train ViT on balanced datasets effectively. Howeverlimited literature discusses the use of ViT for datasets with long-tailedimbalances. In this work we introduce DeiT-LT to tackle the problem oftraining ViTs from scratch on long-tailed datasets. In DeiT-LT we introduce anefficient and effective way of distillation from CNN via distillation DISTtoken by using out-of-distribution images and re-weighting the distillationloss to enhance focus on tail classes. This leads to the learning of localCNN-like features in early ViT blocks improving generalization for tailclasses. Further to mitigate overfitting we propose distilling from a flatCNN teacher which leads to learning low-rank generalizable features for DISTtokens across all ViT blocks. With the proposed DeiT-LT scheme thedistillation DIST token becomes an expert on the tail classes and theclassifier CLS token becomes an expert on the head classes. The experts help toeffectively learn features corresponding to both the majority and minorityclasses using a distinct set of tokens within the same ViT architecture. Weshow the effectiveness of DeiT-LT for training ViT from scratch on datasetsranging from small-scale CIFAR-10 LT to large-scale iNaturalist-2018. |


| Item |Content|
| --- |---|
|idx| 2404.02883v1 |
|title| On the Scalability of Diffusion-based Text-to-Image Generation |
|authors| Hao LiYang ZouYing WangOrchid MajumderYusheng XieR. ManmathaAshwin SwaminathanZhuowen TuStefano ErmonStefano Soatto
|links| http://arxiv.org/abs/2404.02883v1 |
|updated| 2024-04-03 17:34:28 UTC |
|summary| Scaling up model and data size has been quite successful for the evolution ofLLMs. However the scaling law for the diffusion based text-to-image T2Imodels is not fully explored. It is also unclear how to efficiently scale themodel for better performance at reduced cost. The different training settingsand expensive training cost make a fair model comparison extremely difficult.In this work we empirically study the scaling properties of diffusion basedT2I models by performing extensive and rigours ablations on scaling bothdenoising backbones and training set including training scaled UNet andTransformer variants ranging from 0.4B to 4B parameters on datasets upto 600Mimages. For model scaling we find the location and amount of cross attentiondistinguishes the performance of existing UNet designs. And increasing thetransformer blocks is more parameter-efficient for improving text-imagealignment than increasing channel numbers. We then identify an efficient UNetvariant which is 45 smaller and 28 faster than SDXLs UNet. On the datascaling side we show the quality and diversity of the training set mattersmore than simply dataset size. Increasing caption density and diversityimproves text-image alignment performance and the learning efficiency. Finallywe provide scaling functions to predict the text-image alignment performance asfunctions of the scale of model size compute and dataset size. |


| Item |Content|
| --- |---|
|idx| 2404.02877v1 |
|title| FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery |
|authors| Safouane El GhazoualiArnaud GucciardiNicola VenturiMichael RueegseggerUmberto Michelucci
|links| http://arxiv.org/abs/2404.02877v1 |
|updated| 2024-04-03 17:24:27 UTC |
|summary| Object detection in remotely sensed satellite pictures is fundamental in manyfields such as biophysical and environmental monitoring. While deep learningalgorithms are constantly evolving they have been mostly implemented andtested on popular ground-based taken photos. This paper critically evaluatesand compares a suite of advanced object detection algorithms customized for thetask of identifying aircraft within satellite imagery. Using the largeHRPlanesV2 dataset together with a rigorous validation with the GDIT datasetthis research encompasses an array of methodologies including YOLO versions 5and 8 Faster RCNN CenterNet RetinaNet RTMDet and DETR all trained fromscratch. This exhaustive training and validation study reveal YOLOv5 as thepreeminent model for the specific case of identifying airplanes from remotesensing data showcasing high precision and adaptability across diverse imagingconditions. This research highlight the nuanced performance landscapes of thesealgorithms with YOLOv5 emerging as a robust solution for aerial objectdetection underlining its importance through superior mean average precisionRecall and Intersection over Union scores. The findings described hereunderscore the fundamental role of algorithm selection aligned with thespecific demands of satellite imagery analysis and extend a comprehensiveframework to evaluate model efficacy. The benchmark toolkit and codesavailable via https://github.com/toelt-llc/FlightScope_Bench aims to furtherexploration and innovation in the realm of remote sensing object detectionpaving the way for improved analytical methodologies in satellite imageryapplications. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.02904v1 |
|title| ALOHa: A New Measure for Hallucination in Captioning Models |
|authors| Suzanne PetrykDavid M. ChanAnish KachinthayaHaodi ZouJohn CannyJoseph E. GonzalezTrevor Darrell
|links| http://arxiv.org/abs/2404.02904v1 |
|updated| 2024-04-03 17:59:36 UTC |
|summary| Despite recent advances in multimodal pre-training for visual descriptionstate-of-the-art models still produce captions containing errors such ashallucinating objects not present in a scene. The existing prominent metric forobject hallucination CHAIR is limited to a fixed set of MS COCO objects andsynonyms. In this work we propose a modernized open-vocabulary metric ALOHawhich leverages large language models LLMs to measure object hallucinations.Specifically we use an LLM to extract groundable objects from a candidatecaption measure their semantic similarity to reference objects from captionsand object detections and use Hungarian matching to produce a finalhallucination score. We show that ALOHa correctly identifies 13.6 morehallucinated objects than CHAIR on HAT a new gold-standard subset of MS COCOCaptions annotated for hallucinations and 30.8 more on nocaps where objectsextend beyond MS COCO categories. Our code is available athttps://davidmchan.github.io/aloha/. |


| Item |Content|
| --- |---|
|idx| 2404.02900v1 |
|title| DeiT-LT Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets |
|authors| Harsh RangwaniPradipto MondalMayank MishraAshish Ramayee AsokanR. Venkatesh Babu
|links| http://arxiv.org/abs/2404.02900v1 |
|updated| 2024-04-03 17:58:21 UTC |
|summary| Vision Transformer ViT has emerged as a prominent architecture for variouscomputer vision tasks. In ViT we divide the input image into patch tokens andprocess them through a stack of self attention blocks. However unlikeConvolutional Neural Networks CNN ViTs simple architecture has noinformative inductive bias e.g. localityetc. . Due to this ViT requires alarge amount of data for pre-training. Various data efficient approaches DeiThave been proposed to train ViT on balanced datasets effectively. Howeverlimited literature discusses the use of ViT for datasets with long-tailedimbalances. In this work we introduce DeiT-LT to tackle the problem oftraining ViTs from scratch on long-tailed datasets. In DeiT-LT we introduce anefficient and effective way of distillation from CNN via distillation DISTtoken by using out-of-distribution images and re-weighting the distillationloss to enhance focus on tail classes. This leads to the learning of localCNN-like features in early ViT blocks improving generalization for tailclasses. Further to mitigate overfitting we propose distilling from a flatCNN teacher which leads to learning low-rank generalizable features for DISTtokens across all ViT blocks. With the proposed DeiT-LT scheme thedistillation DIST token becomes an expert on the tail classes and theclassifier CLS token becomes an expert on the head classes. The experts help toeffectively learn features corresponding to both the majority and minorityclasses using a distinct set of tokens within the same ViT architecture. Weshow the effectiveness of DeiT-LT for training ViT from scratch on datasetsranging from small-scale CIFAR-10 LT to large-scale iNaturalist-2018. |


| Item |Content|
| --- |---|
|idx| 2404.02896v1 |
|title| Comment on "Machine learning conservation laws from differential equations" |
|authors| Michael F. Zimmer
|links| http://arxiv.org/abs/2404.02896v1 |
|updated| 2024-04-03 17:53:32 UTC |
|summary| In lieu of abstract first paragraph reads: Six months after the authorderived a constant of motion for a 1D damped harmonic oscillator 1 a similarresult appeared by Liu Madhavan and Tegmark 2 3 without citing theauthor. However their derivation contained six serious errors causing boththeir method and result to be incorrect. In this Comment those errors arereviewed. |


| Item |Content|
| --- |---|
|idx| 2404.02892v1 |
|title| MODNO: Multi Operator Learning With Distributed Neural Operators |
|authors| Zecheng Zhang
|links| http://arxiv.org/abs/2404.02892v1 |
|updated| 2024-04-03 17:49:41 UTC |
|summary| The study of operator learning involves the utilization of neural networks toapproximate operators. Traditionally the focus has been on single-operatorlearning SOL. However recent advances have rapidly expanded this to includethe approximation of multiple operators using foundation models equipped withmillions or billions of trainable parameters leading to the research ofmulti-operator learning MOL. In this paper we present a novel distributedtraining approach aimed at enabling a single neural operator with significantlyfewer parameters to effectively tackle multi-operator learning challenges allwithout incurring additional average costs. Our method is applicable to variousChen-Chen-type neural operators such as Deep Operator Neural Networks DON.The core idea is to independently learn the output basis functions for eachoperator using its dedicated data while simultaneously centralizing thelearning of the input function encoding shared by all operators using theentire dataset. Through a systematic study of five numerical examples wecompare the accuracy and cost of training a single neural operator for eachoperator independently versus training a MOL model using our proposed method.Our results demonstrate enhanced efficiency and satisfactory accuracy.Moreover our approach illustrates that some operators with limited data can bemore effectively constructed with the aid of data from analogous operatorsthrough MOL learning. This highlights another MOLs potential to bolsteroperator learning. |


| Item |Content|
| --- |---|
|idx| 2404.02883v1 |
|title| On the Scalability of Diffusion-based Text-to-Image Generation |
|authors| Hao LiYang ZouYing WangOrchid MajumderYusheng XieR. ManmathaAshwin SwaminathanZhuowen TuStefano ErmonStefano Soatto
|links| http://arxiv.org/abs/2404.02883v1 |
|updated| 2024-04-03 17:34:28 UTC |
|summary| Scaling up model and data size has been quite successful for the evolution ofLLMs. However the scaling law for the diffusion based text-to-image T2Imodels is not fully explored. It is also unclear how to efficiently scale themodel for better performance at reduced cost. The different training settingsand expensive training cost make a fair model comparison extremely difficult.In this work we empirically study the scaling properties of diffusion basedT2I models by performing extensive and rigours ablations on scaling bothdenoising backbones and training set including training scaled UNet andTransformer variants ranging from 0.4B to 4B parameters on datasets upto 600Mimages. For model scaling we find the location and amount of cross attentiondistinguishes the performance of existing UNet designs. And increasing thetransformer blocks is more parameter-efficient for improving text-imagealignment than increasing channel numbers. We then identify an efficient UNetvariant which is 45 smaller and 28 faster than SDXLs UNet. On the datascaling side we show the quality and diversity of the training set mattersmore than simply dataset size. Increasing caption density and diversityimproves text-image alignment performance and the learning efficiency. Finallywe provide scaling functions to predict the text-image alignment performance asfunctions of the scale of model size compute and dataset size. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.02905v1 |
|title| Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction |
|authors| Keyu TianYi JiangZehuan YuanBingyue PengLiwei Wang
|links| http://arxiv.org/abs/2404.02905v1 |
|updated| 2024-04-03 17:59:53 UTC |
|summary| We present Visual AutoRegressive modeling VAR a new generation paradigmthat redefines the autoregressive learning on images as coarse-to-finenext-scale prediction or next-resolution prediction diverging from thestandard raster-scan next-token prediction. This simple intuitivemethodology allows autoregressive AR transformers to learn visualdistributions fast and generalize well: VAR for the first time makes ARmodels surpass diffusion transformers in image generation. On ImageNet 256x256benchmark VAR significantly improve AR baseline by improving Frechet inceptiondistance FID from 18.65 to 1.80 inception score IS from 80.4 to 356.4with around 20x faster inference speed. It is also empirically verified thatVAR outperforms the Diffusion Transformer DiT in multiple dimensionsincluding image quality inference speed data efficiency and scalability.Scaling up VAR models exhibits clear power-law scaling laws similar to thoseobserved in LLMs with linear correlation coefficients near -0.998 as solidevidence. VAR further showcases zero-shot generalization ability in downstreamtasks including image in-painting out-painting and editing. These resultssuggest VAR has initially emulated the two important properties of LLMs:Scaling Laws and zero-shot task generalization. We have released all models andcodes to promote the exploration of AR/VAR models for visual generation andunified learning. |


| Item |Content|
| --- |---|
|idx| 2404.02904v1 |
|title| ALOHa: A New Measure for Hallucination in Captioning Models |
|authors| Suzanne PetrykDavid M. ChanAnish KachinthayaHaodi ZouJohn CannyJoseph E. GonzalezTrevor Darrell
|links| http://arxiv.org/abs/2404.02904v1 |
|updated| 2024-04-03 17:59:36 UTC |
|summary| Despite recent advances in multimodal pre-training for visual descriptionstate-of-the-art models still produce captions containing errors such ashallucinating objects not present in a scene. The existing prominent metric forobject hallucination CHAIR is limited to a fixed set of MS COCO objects andsynonyms. In this work we propose a modernized open-vocabulary metric ALOHawhich leverages large language models LLMs to measure object hallucinations.Specifically we use an LLM to extract groundable objects from a candidatecaption measure their semantic similarity to reference objects from captionsand object detections and use Hungarian matching to produce a finalhallucination score. We show that ALOHa correctly identifies 13.6 morehallucinated objects than CHAIR on HAT a new gold-standard subset of MS COCOCaptions annotated for hallucinations and 30.8 more on nocaps where objectsextend beyond MS COCO categories. Our code is available athttps://davidmchan.github.io/aloha/. |


| Item |Content|
| --- |---|
|idx| 2404.02903v1 |
|title| LidarDM: Generative LiDAR Simulation in a Generated World |
|authors| Vlas ZyrianovHenry CheZhijian LiuShenlong Wang
|links| http://arxiv.org/abs/2404.02903v1 |
|updated| 2024-04-03 17:59:28 UTC |
|summary| We present LidarDM a novel LiDAR generative model capable of producingrealistic layout-aware physically plausible and temporally coherent LiDARvideos. LidarDM stands out with two unprecedented capabilities in LiDARgenerative modeling: i LiDAR generation guided by driving scenarios offeringsignificant potential for autonomous driving simulations and ii 4D LiDARpoint cloud generation enabling the creation of realistic and temporallycoherent sequences. At the heart of our model is a novel integrated 4D worldgeneration framework. Specifically we employ latent diffusion models togenerate the 3D scene combine it with dynamic actors to form the underlying 4Dworld and subsequently produce realistic sensory observations within thisvirtual environment. Our experiments indicate that our approach outperformscompeting algorithms in realism temporal coherency and layout consistency. Weadditionally show that LidarDM can be used as a generative world modelsimulator for training and testing perception models. |


| Item |Content|
| --- |---|
|idx| 2404.02900v1 |
|title| DeiT-LT Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets |
|authors| Harsh RangwaniPradipto MondalMayank MishraAshish Ramayee AsokanR. Venkatesh Babu
|links| http://arxiv.org/abs/2404.02900v1 |
|updated| 2024-04-03 17:58:21 UTC |
|summary| Vision Transformer ViT has emerged as a prominent architecture for variouscomputer vision tasks. In ViT we divide the input image into patch tokens andprocess them through a stack of self attention blocks. However unlikeConvolutional Neural Networks CNN ViTs simple architecture has noinformative inductive bias e.g. localityetc. . Due to this ViT requires alarge amount of data for pre-training. Various data efficient approaches DeiThave been proposed to train ViT on balanced datasets effectively. Howeverlimited literature discusses the use of ViT for datasets with long-tailedimbalances. In this work we introduce DeiT-LT to tackle the problem oftraining ViTs from scratch on long-tailed datasets. In DeiT-LT we introduce anefficient and effective way of distillation from CNN via distillation DISTtoken by using out-of-distribution images and re-weighting the distillationloss to enhance focus on tail classes. This leads to the learning of localCNN-like features in early ViT blocks improving generalization for tailclasses. Further to mitigate overfitting we propose distilling from a flatCNN teacher which leads to learning low-rank generalizable features for DISTtokens across all ViT blocks. With the proposed DeiT-LT scheme thedistillation DIST token becomes an expert on the tail classes and theclassifier CLS token becomes an expert on the head classes. The experts help toeffectively learn features corresponding to both the majority and minorityclasses using a distinct set of tokens within the same ViT architecture. Weshow the effectiveness of DeiT-LT for training ViT from scratch on datasetsranging from small-scale CIFAR-10 LT to large-scale iNaturalist-2018. |


| Item |Content|
| --- |---|
|idx| 2404.02899v1 |
|title| MatAtlas: Text-driven Consistent Geometry Texturing and Material Assignment |
|authors| Duygu CeylanValentin DeschaintreThibault GroueixRosalie MartinChun-Hao HuangRomain RouffetVladimir KimGaëtan Lassagne
|links| http://arxiv.org/abs/2404.02899v1 |
|updated| 2024-04-03 17:57:15 UTC |
|summary| We present MatAtlas a method for consistent text-guided 3D model texturing.Following recent progress we leverage a large scale text-to-image generationmodel e.g. Stable Diffusion as a prior to texture a 3D model. We carefullydesign an RGB texturing pipeline that leverages a grid pattern diffusiondriven by depth and edges. By proposing a multi-step texture refinementprocess we significantly improve the quality and 3D consistency of thetexturing output. To further address the problem of baked-in lighting we movebeyond RGB colors and pursue assigning parametric materials to the assets.Given the high-quality initial RGB texture we propose a novel materialretrieval method capitalized on Large Language Models LLM enablingeditabiliy and relightability. We evaluate our method on a wide variety ofgeometries and show that our method significantly outperform prior arts. Wealso analyze the role of each component through a detailed ablation study. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.02873v1 |
|title| Gaussian Process Regression with Soft Inequality and Monotonicity Constraints |
|authors| Didem KochanXiu Yang
|links| http://arxiv.org/abs/2404.02873v1 |
|updated| 2024-04-03 17:09:25 UTC |
|summary| Gaussian process GP regression is a non-parametric Bayesian framework toapproximate complex models. Standard GP regression can lead to an unboundedmodel in which some points can take infeasible values. We introduce a new GPmethod that enforces the physical constraints in a probabilistic manner. ThisGP model is trained by the quantum-inspired Hamiltonian Monte Carlo QHMC.QHMC is an efficient way to sample from a broad class of distributions. Unlikethe standard Hamiltonian Monte Carlo algorithm in which a particle has a fixedmass QHMC allows a particle to have a random mass matrix with a probabilitydistribution. Introducing the QHMC method to the inequality and monotonicityconstrained GP regression in the probabilistic sense our approach improves theaccuracy and reduces the variance in the resulting GP model. According to ourexperiments on several datasets the proposed approach serves as an efficientmethod as it accelerates the sampling process while maintaining the accuracyand it is applicable to high dimensional problems. |


| Item |Content|
| --- |---|
|idx| 2404.02866v1 |
|title| Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds |
|authors| Kamalika ChaudhuriChuan GuoLaurens van der MaatenSaeed MahloujifarMark Tygert
|links| http://arxiv.org/abs/2404.02866v1 |
|updated| 2024-04-03 16:58:03 UTC |
|summary| Protecting privacy during inference with deep neural networks is possible byadding noise to the activations in the last layers prior to the finalclassifiers or other task-specific layers. The activations in such layers areknown as features or less commonly as embeddings or featureembeddings. The added noise helps prevent reconstruction of the inputs fromthe noisy features. Lower bounding the variance of every possible unbiasedestimator of the inputs quantifies the confidentiality arising from such addednoise. Convenient computationally tractable bounds are available from classicinequalities of Hammersley and of Chapman and Robbins -- the HCR bounds.Numerical experiments indicate that the HCR bounds are on the precipice ofbeing effectual for small neural nets with the data sets MNIST andCIFAR-10 which contain 10 classes each for image classification. The HCRbounds appear to be insufficient on their own to guarantee confidentiality ofthe inputs to inference with standard deep neural nets ResNet-18 andSwin-T pre-trained on the data set ImageNet-1000 which contains 1000classes. Supplementing the addition of noise to features with other methods forproviding confidentiality may be warranted in the case of ImageNet. In allcases the results reported here limit consideration to amounts of added noisethat incur little degradation in the accuracy of classification from the noisyfeatures. Thus the added noise enhances confidentiality without much reductionin the accuracy on the task of image classification. |


| Item |Content|
| --- |---|
|idx| 2404.02591v1 |
|title| Adaptive Sampling Policies Imply Biased Beliefs: A Generalization of the Hot Stove Effect |
|authors| Jerker Denrell
|links| http://arxiv.org/abs/2404.02591v1 |
|updated| 2024-04-03 09:15:38 UTC |
|summary| The Hot Stove Effect is a negativity bias resulting from the adaptivecharacter of learning. The mechanism is that learning algorithms that pursuealternatives with positive estimated values but avoid alternatives withnegative estimated values will correct errors of overestimation but fail tocorrect errors of underestimation. Here we generalize the theory behind theHot Stove Effect to settings in which negative estimates do not necessarilylead to avoidance but to a smaller sample size i.e. a learner selects fewerof alternative B if B is believed to be inferior but does not entirely avoidB. We formally demonstrate that the negativity bias remains in this set-up. Wealso show there is a negativity bias for Bayesian learners in the sense thatmost such learners underestimate the expected value of an alternative. |


| Item |Content|
| --- |---|
|idx| 2404.02538v1 |
|title| Convergence Analysis of Flow Matching in Latent Space with Transformers |
|authors| Yuling JiaoYanming LaiYang WangBokai Yan
|links| http://arxiv.org/abs/2404.02538v1 |
|updated| 2024-04-03 07:50:53 UTC |
|summary| We present theoretical convergence guarantees for ODE-based generativemodels specifically flow matching. We use a pre-trained autoencoder network tomap high-dimensional original inputs to a low-dimensional latent space where atransformer network is trained to predict the velocity field of thetransformation from a standard normal distribution to the target latentdistribution. Our error analysis demonstrates the effectiveness of thisapproach showing that the distribution of samples generated via estimated ODEflow converges to the target distribution in the Wasserstein-2 distance undermild and practical assumptions. Furthermore we show that arbitrary smoothfunctions can be effectively approximated by transformer networks withLipschitz continuity which may be of independent interest. |


| Item |Content|
| --- |---|
|idx| 2404.02446v1 |
|title| Masked Completion via Structured Diffusion with White-Box Transformers |
|authors| Druv PaiZiyang WuSam BuchananYaodong YuYi Ma
|links| http://arxiv.org/abs/2404.02446v1 |
|updated| 2024-04-03 04:23:01 UTC |
|summary| Modern learning frameworks often train deep neural networks with massiveamounts of unlabeled data to learn representations by solving simple pretexttasks then use the representations as foundations for downstream tasks. Thesenetworks are empirically designed as such they are usually not interpretabletheir representations are not structured and their designs are potentiallyredundant. White-box deep networks in which each layer explicitly identifiesand transforms structures in the data present a promising alternative.However existing white-box architectures have only been shown to work at scalein supervised settings with labeled data such as classification. In this workwe provide the first instantiation of the white-box design paradigm that can beapplied to large-scale unsupervised representation learning. We do this byexploiting a fundamental connection between diffusion compression andmasked completion deriving a deep transformer-like masked autoencoderarchitecture called CRATE-MAE in which the role of each layer ismathematically fully interpretable: they transform the data distribution to andfrom a structured representation. Extensive empirical evaluations confirm ouranalytical insights. CRATE-MAE demonstrates highly promising performance onlarge-scale imagery datasets while using only 30 of the parameters comparedto the standard masked autoencoder with the same model configuration. Therepresentations learned by CRATE-MAE have explicit structure and also containsemantic meaning. Code is available at https://github.com/Ma-Lab-Berkeley/CRATE . |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.02880v1 |
|title| Fragmented Moments, Balanced Choices: How Do People Make Use of Their Waiting Time? |
|authors| Jian ZhengGe Gao
|links| http://dx.doi.org/10.1145/3613904.3642608 |
|updated| 2024-04-03 17:32:50 UTC |
|summary| Everyone spends some time waiting every day. HCI research has developed toolsfor boosting productivity while waiting. However little is known about howpeople naturally spend their waiting time. We conducted an experience samplingstudy with 21 working adults who used a mobile app to report their dailywaiting time activities over two weeks. The aim of this study is to understandthe activities people do while waiting and the effect of situational factors.We found that participants spent about 60 of their waiting time on leisureactivities 20 on productive activities and 20 on maintenance activities.These choices are sensitive to situational factors including accessibledevice location and certain routines of the day. Our study complementsprevious ones by demonstrating that people purpose waiting time for variousgoals beyond productivity and to maintain work-life balance. Our findings shedlight on future empirical research and system design for time management. |


| Item |Content|
| --- |---|
|idx| 2404.02806v1 |
|title| The RealHumanEval: Evaluating Large Language Models' Abilities to Support Programmers |
|authors| Hussein MozannarValerie ChenMohammed AlsobaySubhro DasSebastian ZhaoDennis WeiManish NagireddyPrasanna SattigeriAmeet TalwalkarDavid Sontag
|links| http://arxiv.org/abs/2404.02806v1 |
|updated| 2024-04-03 15:20:57 UTC |
|summary| Evaluation of large language models LLMs for code has primarily relied onstatic benchmarks including HumanEval Chen et al. 2021 which measure theability of LLMs to generate complete code that passes unit tests. As LLMs areincreasingly used as programmer assistants we study whether gains on existingbenchmarks translate to gains in programmer productivity when coding with LLMsincluding time spent coding. In addition to static benchmarks we investigatethe utility of preference metrics that might be used as proxies to measure LLMhelpfulness such as code acceptance or copy rates. To do so we introduceRealHumanEval a web interface to measure the ability of LLMs to assistprogrammers through either autocomplete or chat support. We conducted a userstudy N213 using RealHumanEval in which users interacted with six LLMs ofvarying base model performance. Despite static benchmarks not incorporatinghumans-in-the-loop we find that improvements in benchmark performance lead toincreased programmer productivity however gaps in benchmark versus humanperformance are not proportional -- a trend that holds across both forms of LLMsupport. In contrast we find that programmer preferences do not correlate withtheir actual performance motivating the need for better human-centric proxysignals. We also open-source RealHumanEval to enable human-centric evaluationof new models and the study data to facilitate efforts to improve code models. |


| Item |Content|
| --- |---|
|idx| 2404.02798v1 |
|title| AI and personalized learning: bridging the gap with modern educational goals |
|authors| Kristjan-Julius LaakJaan Aru
|links| http://arxiv.org/abs/2404.02798v1 |
|updated| 2024-04-03 15:07:00 UTC |
|summary| Personalized learning PL aspires to provide an alternative to theone-size-fits-all approach in education. Technology-based PL solutions haveshown notable effectiveness in enhancing learning performance. However theiralignment with the broader goals of modern education is inconsistent acrosstechnologies and research areas. In this paper we examine the characteristicsof AI-driven PL solutions in light of the OECD Learning Compass 2030 goals. Ouranalysis indicates a gap between the objectives of modern education and thecurrent direction of PL. We identify areas where most present-day PLtechnologies could better embrace essential elements of contemporary educationsuch as collaboration cognitive engagement and the development of generalcompetencies. While the present PL solutions are instrumental in aidinglearning processes the PL envisioned by educational experts extends beyondsimple technological tools and requires a holistic change in the educationalsystem. Finally we explore the potential of large language models such asChatGPT and propose a hybrid model that blends artificial intelligence with acollaborative teacher-facilitated approach to personalized learning. |


| Item |Content|
| --- |---|
|idx| 2404.02743v1 |
|title| IEEE VIS Workshop on Visualization for Climate Action and Sustainability |
|authors| Benjamin BachFanny ChevalierHelen-Nicole KostisMark SubbaroYvonne JansenRobert Soden
|links| http://arxiv.org/abs/2404.02743v1 |
|updated| 2024-04-03 13:39:59 UTC |
|summary| This first workshop on visualization for climate action and sustainabilityaims to explore and consolidate the role of data visualization in acceleratingaction towards addressing the current environmental crisis. Given the urgencyand impact of the environmental crisis we ask how our skills researchmethods and innovations can help by empowering people and organizations. Webelieve visualization holds an enormous power to aid understanding decisionmaking communication discussion participation education and exploration ofcomplex topics around climate action and sustainability. Hence this workshopinvites submissions and discussion around these topics with the goal ofestablishing a visible and actionable link between these fields and theirrespective stakeholders. The workshop solicits work-in-progress and researchpapers as well as pictorials and interactive demos from the whole range ofvisualization research dashboards interactive spaces scientificvisualization storytelling visual analytics explainability etc. within thecontext of environmentalism climate science sustainability energy circulareconomy biodiversity etc. and across a range of scenarios from publicawareness and understanding visual analysis expert decision making sciencecommunication personal decision making etc. After presentations ofsubmissions the workshop will feature dedicated discussion groups around datadriven interactive experiences for the public and tools for personal andprofessional decision making. |


| Item |Content|
| --- |---|
|idx| 2404.02718v1 |
|title| Evolving Agents: Interactive Simulation of Dynamic and Diverse Human Personalities |
|authors| Jiale LiJiayang LiJiahao ChenYifan LiShijie WangHugo ZhouMinjun YeYunsheng Su
|links| http://arxiv.org/abs/2404.02718v1 |
|updated| 2024-04-03 13:20:36 UTC |
|summary| Human-like Agents with diverse and dynamic personality could serve as animportant design probe in the process of user-centered design thereby enablingdesigners to enhance the user experience of interactive application.In thisarticle we introduce Evolving Agents a novel agent architecture that consistsof two systems: Personality and Behavior. The Personality system includes threemodules: Cognition Emotion and Character Growth. The Behavior system comprisestwo modules: Planning and Action. We also build a simulation platform thatenables agents to interact with the environment and other agents. EvolvingAgents can simulate the human personality evolution process. Compared to itsinitial state agents personality and behavior patterns undergo believabledevelopment after several days of simulation. Agents reflect on their behaviorto reason and develop new personality traits. These traits in turn generatenew behavior patterns forming a feedback loop-like personality evolution.Inour experiment we utilized simulation platform with 10 agents for evaluation.During the evaluation these agents experienced believable and inspirationalpersonality evolution. Through ablation and control experiments wedemonstrated the outstanding effectiveness of agent personality evolution andall modules of our agent architecture contribute to creating believablehuman-like agents with diverse and dynamic personalities. We also demonstratedthrough workshops how Evolving Agents could inspire designers. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.02448v1 |
|title| Electric Vehicle Routing Problem for Emergency Power Supply: Towards Telecom Base Station Relief |
|authors| Daisuke KikutaHiroki IkeuchiKengo TajiriYuta ToyamaYuusuke Nakano
|links| http://arxiv.org/abs/2404.02448v1 |
|updated| 2024-04-03 04:27:07 UTC |
|summary| As a telecom provider our company has a critical mission to maintain telecomservices even during power outages. To accomplish the mission it is essentialto maintain the power of the telecom base stations. Here we consider a solutionwhere electric vehicles EVs directly supply power to base stations bytraveling to their locations. The goal is to find EV routes that minimize boththe total travel distance of all EVs and the number of downed base stations. Inthis paper we formulate this routing problem as a new variant of the ElectricVehicle Routing Problem EVRP and propose a solver that combines a rule-basedvehicle selector and a reinforcement learning RL-based node selector. Therule of the vehicle selector ensures the exact environmental states when theselected EV starts to move. In addition the node selection by the RL modelenables fast route generation which is critical in emergencies. We evaluateour solver on both synthetic datasets and real datasets. The results show thatour solver outperforms baselines in terms of the objective value andcomputation time. Moreover we analyze the generalization and scalability ofour solver demonstrating the capability toward unseen settings and large-scaleproblems. Check also our project page: https://ntt-dkiku.github.io/rl-evrpeps. |


| Item |Content|
| --- |---|
|idx| 2404.02361v1 |
|title| EnergAIze: Multi Agent Deep Deterministic Policy Gradient for Vehicle to Grid Energy Management |
|authors| Tiago FonsecaLuis FerreiraBernardo CabralRicardo SeverinoIsabel Praca
|links| http://arxiv.org/abs/2404.02361v1 |
|updated| 2024-04-02 23:16:17 UTC |
|summary| This paper investigates the increasing roles of Renewable Energy SourcesRES and Electric Vehicles EVs. While indicating a new era of sustainableenergy these also introduce complex challenges including the need to balancesupply and demand and smooth peak consumptions amidst rising EV adoption rates.Addressing these challenges requires innovative solutions such as DemandResponse DR energy flexibility management Renewable Energy CommunitiesRECs and more specifically for EVs Vehicle-to-Grid V2G. However existingV2G approaches often fall short in real-world adaptability global RECoptimization with other flexible assets scalability and user engagement. Tobridge this gap this paper introduces EnergAIze a Multi-Agent ReinforcementLearning MARL energy management framework leveraging the Multi-Agent DeepDeterministic Policy Gradient MADDPG algorithm. EnergAIze enablesuser-centric and multi-objective energy management by allowing each prosumer toselect from a range of personal management objectives thus encouragingengagement. Additionally it architects data protection and ownership throughdecentralized computing where each prosumer can situate an energy managementoptimization node directly at their own dwelling. The local node not onlymanages local energy assets but also fosters REC wide optimization. Theefficacy of EnergAIze was evaluated through case studies employing theCityLearn simulation framework. These simulations were instrumental indemonstrating EnergAIzes adeptness at implementing V2G technology within a RECand other energy assets. The results show reduction in peak loads rampingcarbon emissions and electricity costs at the REC level while optimizing forindividual prosumers objectives. |


| Item |Content|
| --- |---|
|idx| 2404.02289v1 |
|title| Federated Multi-Agent Mapping for Planetary Exploration |
|authors| Tiberiu-Ioan SzatmariAbhishek Cauligi
|links| http://arxiv.org/abs/2404.02289v1 |
|updated| 2024-04-02 20:32:32 UTC |
|summary| In multi-agent robotic exploration managing and effectively utilizing thevast heterogeneous data generated from dynamic environments poses asignificant challenge. Federated learning FL is a promising approach fordistributed mapping addressing the challenges of decentralized data incollaborative learning. FL enables joint model training across multiple agentswithout requiring the centralization or sharing of raw data overcomingbandwidth and storage constraints. Our approach leverages implicit neuralmapping representing maps as continuous functions learned by neural networksfor compact and adaptable representations. We further enhance this approachwith meta-initialization on Earth datasets pre-training the network to quicklylearn new map structures. This combination demonstrates strong generalizationto diverse domains like Martian terrain and glaciers. We rigorously evaluatethis approach demonstrating its effectiveness for real-world deployment inmulti-agent exploration scenarios. |


| Item |Content|
| --- |---|
|idx| 2404.01999v1 |
|title| Emergence of Chemotactic Strategies with Multi-Agent Reinforcement Learning |
|authors| Samuel ToveyChristoph LohrmannChristian Holm
|links| http://arxiv.org/abs/2404.01999v1 |
|updated| 2024-04-02 14:42:52 UTC |
|summary| Reinforcement learning RL is a flexible and efficient method forprogramming micro-robots in complex environments. Here we investigate whetherreinforcement learning can provide insights into biological systems whentrained to perform chemotaxis. Namely whether we can learn about howintelligent agents process given information in order to swim towards a target.We run simulations covering a range of agent shapes sizes and swim speeds todetermine if the physical constraints on biological swimmers namely Brownianmotion lead to regions where reinforcement learners training fails. We findthat the RL agents can perform chemotaxis as soon as it is physically possibleand in some cases even before the active swimming overpowers the stochasticenvironment. We study the efficiency of the emergent policy and identifyconvergence in agent size and swim speeds. Finally we study the strategyadopted by the reinforcement learning algorithm to explain how the agentsperform their tasks. To this end we identify three emerging dominantstrategies and several rare approaches taken. These strategies whilstproducing almost identical trajectories in simulation are distinct and giveinsight into the possible mechanisms behind which biological agents exploretheir environment and respond to changing conditions. |


| Item |Content|
| --- |---|
|idx| 2404.02183v1 |
|title| Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization |
|authors| Yoichi IshibashiYoshimasa Nishimura
|links| http://arxiv.org/abs/2404.02183v1 |
|updated| 2024-04-02 13:37:28 UTC |
|summary| Recent advancements in automatic code generation using large language modelLLM agent have brought us closer to the future of automated softwaredevelopment. However existing single-agent approaches face limitations ingenerating and improving large-scale complex codebases due to constraints incontext length. To tackle this challenge we propose Self-Organized multi-Agentframework SoA a novel multi-agent framework that enables the scalable andefficient generation and optimization of large-scale code. In SoAself-organized agents operate independently to generate and modify codecomponents while seamlessly collaborating to construct the overall codebase. Akey feature of our framework is the automatic multiplication of agents based onproblem complexity allowing for dynamic scalability. This enables the overallcode volume to be increased indefinitely according to the number of agentswhile the amount of code managed by each agent remains constant. We evaluateSoA on the HumanEval benchmark and demonstrate that compared to a single-agentsystem each agent in SoA handles significantly less code yet the overallgenerated code is substantially greater. Moreover SoA surpasses the powerfulsingle-agent baseline by 5 in terms of Pass1 accuracy. |


