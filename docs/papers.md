# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.03496v1 |
|title| Wings: Learning Multimodal LLMs without Text-only Forgetting |
|authors| Yi-Kai ZhangShiyin LuYang LiYanqing MaQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.03496v1 |
|updated| 2024-06-05 17:59:40 UTC |
|summary| Multimodal large language models MLLMs initiated with a trained LLM firstalign images with text and then fine-tune on multimodal mixed inputs. Howeverthe MLLM catastrophically forgets the text-only instructions which do notinclude images and can be addressed within the initial LLM. In this paper wepresent Wings a novel MLLM that excels in both text-only dialogues andmultimodal comprehension. Analyzing MLLM attention in multimodal instructionsreveals that text-only forgetting is related to the attention shifts frompre-image to post-image text. From that we construct extra modules that act asthe boosted learner to compensate for the attention shift. The complementaryvisual and textual learners like wings on either side are connected inparallel within each layers attention block. Initially image and text inputsare aligned with visual learners operating alongside the main attentionbalancing focus on visual elements. Textual learners are later collaborativelyintegrated with attention-based routing to blend the outputs of the visual andtextual learners. We design the Low-Rank Residual Attention LoRRA toguarantee high efficiency for learners. Our experimental results demonstratethat Wings outperforms equally-scaled MLLMs in both text-only and visualquestion-answering tasks. On a newly constructed Interleaved Image-Text IITbenchmark Wings exhibits superior performance from text-only-rich tomultimodal-rich question-answering tasks. |


| Item |Content|
| --- |---|
|idx| 2406.03487v1 |
|title| Analyzing LLM Behavior in Dialogue Summarization: Unveiling Circumstantial Hallucination Trends |
|authors| Sanjana RamprasadElisa FerracaneZachary C. Lipton
|links| http://arxiv.org/abs/2406.03487v1 |
|updated| 2024-06-05 17:49:47 UTC |
|summary| Recent advancements in large language models LLMs have considerablyadvanced the capabilities of summarization systems. However they continue toface concerns about hallucinations. While prior work has evaluated LLMsextensively in news domains most evaluation of dialogue summarization hasfocused on BART-based models leaving a gap in our understanding of theirfaithfulness. Our work benchmarks the faithfulness of LLMs for dialoguesummarization using human annotations and focusing on identifying andcategorizing span-level inconsistencies. Specifically we focus on twoprominent LLMs: GPT-4 and Alpaca-13B. Our evaluation reveals subtleties as towhat constitutes a hallucination: LLMs often generate plausible inferencessupported by circumstantial evidence in the conversation that lack directevidence a pattern that is less prevalent in older models. We propose arefined taxonomy of errors coining the category of Circumstantial Inferenceto bucket these LLM behaviors and release the dataset. Using our taxonomy wecompare the behavioral differences between LLMs and older fine-tuned models.Additionally we systematically assess the efficacy of automatic errordetection methods on LLM summaries and find that they struggle to detect thesenuanced errors. To address this we introduce two prompt-based approaches forfine-grained error detection that outperform existing metrics particularly foridentifying Circumstantial Inference. |


| Item |Content|
| --- |---|
|idx| 2406.03486v1 |
|title| BIPED: Pedagogically Informed Tutoring System for ESL Education |
|authors| Soonwoo KwonSojung KimMinju ParkSeunghyun LeeKyuseok Kim
|links| http://arxiv.org/abs/2406.03486v1 |
|updated| 2024-06-05 17:49:24 UTC |
|summary| Large Language Models LLMs have a great potential to serve as readilyavailable and cost-efficient Conversational Intelligent Tutoring Systems CITSfor teaching L2 learners of English. Existing CITS however are designed toteach only simple concepts or lack the pedagogical depth necessary to addressdiverse learning strategies. To develop a more pedagogically informed CITScapable of teaching complex concepts we construct a BIlingualPEDagogically-informed Tutoring Dataset BIPED of one-on-one human-to-humanEnglish tutoring interactions. Through post-hoc analysis of the tutoringinteractions we come up with a lexicon of dialogue acts 34 tutor acts and 9student acts which we use to further annotate the collected dataset. Based ona two-step framework of first predicting the appropriate tutor act thengenerating the corresponding response we implemented two CITS models usingGPT-4 and SOLAR-KO respectively. We experimentally demonstrate that theimplemented models not only replicate the style of human teachers but alsoemploy diverse and contextually appropriate pedagogical strategies. |


| Item |Content|
| --- |---|
|idx| 2406.03482v1 |
|title| QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead |
|authors| Amir ZandiehMajid DaliriInsu Han
|links| http://arxiv.org/abs/2406.03482v1 |
|updated| 2024-06-05 17:42:05 UTC |
|summary| Serving LLMs requires substantial memory due to the storage requirements ofKey-Value KV embeddings in the KV cache which grows with sequence length. Aneffective approach to compress KV cache is quantization. However traditionalquantization methods face significant memory overhead due to the need to storequantization constants at least a zero point and a scale in full precisionper data block. Depending on the block size this overhead can add 1 or 2 bitsper quantized number. We introduce QJL a new quantization approach thatconsists of a Johnson-Lindenstrauss JL transform followed by sign-bitquantization. In contrast to existing methods QJL eliminates memory overheadsby removing the need for storing quantization constants. We propose anasymmetric estimator for the inner product of two vectors and demonstrate thatapplying QJL to one vector and a standard JL transform without quantization tothe other provides an unbiased estimator with minimal distortion. We havedeveloped an efficient implementation of the QJL sketch and its correspondinginner product estimator incorporating a lightweight CUDA kernel for optimizedcomputation. When applied across various LLMs and NLP tasks to quantize the KVcache to only 3 bits QJL demonstrates a more than fivefold reduction in KVcache memory usage without compromising accuracy all while achieving fasterruntime. Codes are available at urlhttps://github.com/amirzandieh/QJL. |


| Item |Content|
| --- |---|
|idx| 2406.03479v1 |
|title| MODABS: Multi-Objective Learning for Dynamic Aspect-Based Summarization |
|authors| Xiaobo GuoSoroush Vosoughi
|links| http://arxiv.org/abs/2406.03479v1 |
|updated| 2024-06-05 17:32:28 UTC |
|summary| The rapid proliferation of online content necessitates effectivesummarization methods among which dynamic aspect-based summarization standsout. Unlike its traditional counterpart which assumes a fixed set of knownaspects this approach adapts to the varied aspects of the input text. Weintroduce a novel multi-objective learning framework employing aLongformer-Encoder-Decoder for this task. The framework optimizes aspect numberprediction minimizes disparity between generated and reference summaries foreach aspect and maximizes dissimilarity across aspect-specific summaries.Extensive experiments show our method significantly outperforms baselines onthree diverse datasets largely due to the effective alignment of generated andreference aspect counts without sacrificing single-aspect summarizationquality. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.03496v1 |
|title| Wings: Learning Multimodal LLMs without Text-only Forgetting |
|authors| Yi-Kai ZhangShiyin LuYang LiYanqing MaQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.03496v1 |
|updated| 2024-06-05 17:59:40 UTC |
|summary| Multimodal large language models MLLMs initiated with a trained LLM firstalign images with text and then fine-tune on multimodal mixed inputs. Howeverthe MLLM catastrophically forgets the text-only instructions which do notinclude images and can be addressed within the initial LLM. In this paper wepresent Wings a novel MLLM that excels in both text-only dialogues andmultimodal comprehension. Analyzing MLLM attention in multimodal instructionsreveals that text-only forgetting is related to the attention shifts frompre-image to post-image text. From that we construct extra modules that act asthe boosted learner to compensate for the attention shift. The complementaryvisual and textual learners like wings on either side are connected inparallel within each layers attention block. Initially image and text inputsare aligned with visual learners operating alongside the main attentionbalancing focus on visual elements. Textual learners are later collaborativelyintegrated with attention-based routing to blend the outputs of the visual andtextual learners. We design the Low-Rank Residual Attention LoRRA toguarantee high efficiency for learners. Our experimental results demonstratethat Wings outperforms equally-scaled MLLMs in both text-only and visualquestion-answering tasks. On a newly constructed Interleaved Image-Text IITbenchmark Wings exhibits superior performance from text-only-rich tomultimodal-rich question-answering tasks. |


| Item |Content|
| --- |---|
|idx| 2406.03487v1 |
|title| Analyzing LLM Behavior in Dialogue Summarization: Unveiling Circumstantial Hallucination Trends |
|authors| Sanjana RamprasadElisa FerracaneZachary C. Lipton
|links| http://arxiv.org/abs/2406.03487v1 |
|updated| 2024-06-05 17:49:47 UTC |
|summary| Recent advancements in large language models LLMs have considerablyadvanced the capabilities of summarization systems. However they continue toface concerns about hallucinations. While prior work has evaluated LLMsextensively in news domains most evaluation of dialogue summarization hasfocused on BART-based models leaving a gap in our understanding of theirfaithfulness. Our work benchmarks the faithfulness of LLMs for dialoguesummarization using human annotations and focusing on identifying andcategorizing span-level inconsistencies. Specifically we focus on twoprominent LLMs: GPT-4 and Alpaca-13B. Our evaluation reveals subtleties as towhat constitutes a hallucination: LLMs often generate plausible inferencessupported by circumstantial evidence in the conversation that lack directevidence a pattern that is less prevalent in older models. We propose arefined taxonomy of errors coining the category of Circumstantial Inferenceto bucket these LLM behaviors and release the dataset. Using our taxonomy wecompare the behavioral differences between LLMs and older fine-tuned models.Additionally we systematically assess the efficacy of automatic errordetection methods on LLM summaries and find that they struggle to detect thesenuanced errors. To address this we introduce two prompt-based approaches forfine-grained error detection that outperform existing metrics particularly foridentifying Circumstantial Inference. |


| Item |Content|
| --- |---|
|idx| 2406.03485v1 |
|title| Highway Value Iteration Networks |
|authors| Yuhui WangWeida LiFrancesco FaccioQingyuan WuJürgen Schmidhuber
|links| http://arxiv.org/abs/2406.03485v1 |
|updated| 2024-06-05 17:46:26 UTC |
|summary| Value iteration networks VINs enable end-to-end learning for planning tasksby employing a differentiable planning module that approximates the valueiteration algorithm. However long-term planning remains a challenge becausetraining very deep VINs is difficult. To address this problem we embed highwayvalue iteration -- a recent algorithm designed to facilitate long-term creditassignment -- into the structure of VINs. This improvement augments theplanning module of the VIN with three additional components: 1 an aggregategate which constructs skip connections to improve information flow acrossmany layers 2 an exploration module crafted to increase the diversity ofinformation and gradient flow in spatial dimensions 3 a filter gatedesigned to ensure safe exploration. The resulting novel highway VIN can betrained effectively with hundreds of layers using standard backpropagation. Inlong-term planning tasks requiring hundreds of planning steps deep highwayVINs outperform both traditional VINs and several advanced very deep NNs. |


| Item |Content|
| --- |---|
|idx| 2406.03482v1 |
|title| QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead |
|authors| Amir ZandiehMajid DaliriInsu Han
|links| http://arxiv.org/abs/2406.03482v1 |
|updated| 2024-06-05 17:42:05 UTC |
|summary| Serving LLMs requires substantial memory due to the storage requirements ofKey-Value KV embeddings in the KV cache which grows with sequence length. Aneffective approach to compress KV cache is quantization. However traditionalquantization methods face significant memory overhead due to the need to storequantization constants at least a zero point and a scale in full precisionper data block. Depending on the block size this overhead can add 1 or 2 bitsper quantized number. We introduce QJL a new quantization approach thatconsists of a Johnson-Lindenstrauss JL transform followed by sign-bitquantization. In contrast to existing methods QJL eliminates memory overheadsby removing the need for storing quantization constants. We propose anasymmetric estimator for the inner product of two vectors and demonstrate thatapplying QJL to one vector and a standard JL transform without quantization tothe other provides an unbiased estimator with minimal distortion. We havedeveloped an efficient implementation of the QJL sketch and its correspondinginner product estimator incorporating a lightweight CUDA kernel for optimizedcomputation. When applied across various LLMs and NLP tasks to quantize the KVcache to only 3 bits QJL demonstrates a more than fivefold reduction in KVcache memory usage without compromising accuracy all while achieving fasterruntime. Codes are available at urlhttps://github.com/amirzandieh/QJL. |


| Item |Content|
| --- |---|
|idx| 2406.03450v1 |
|title| What is the Best Way for ChatGPT to Translate Poetry? |
|authors| Shanshan WangDerek F. WongJingming YaoLidia S. Chao
|links| http://arxiv.org/abs/2406.03450v1 |
|updated| 2024-06-05 16:48:26 UTC |
|summary| Machine translation MT has historically faced significant challenges whenapplied to literary works particularly in the domain of poetry translation.The advent of Large Language Models such as ChatGPT holds potential forinnovation in this field. This study examines ChatGPTs capabilities inEnglish-Chinese poetry translation tasks utilizing targeted prompts and smallsample scenarios to ascertain optimal performance. Despite promising outcomesour analysis reveals persistent issues in the translations generated by ChatGPTthat warrant attention. To address these shortcomings we propose anExplanation-Assisted Poetry Machine Translation EAPMT method which leveragesmonolingual poetry explanation as a guiding information for the translationprocess. Furthermore we refine existing evaluation criteria to better suit thenuances of modern poetry translation. We engaged a panel of professional poetsfor assessments complemented evaluations by using GPT-4. The results from bothhuman and machine evaluations demonstrate that our EAPMT method outperformstraditional translation methods of ChatGPT and the existing online systems.This paper validates the efficacy of our method and contributes a novelperspective to machine-assisted literary translation. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.03496v1 |
|title| Wings: Learning Multimodal LLMs without Text-only Forgetting |
|authors| Yi-Kai ZhangShiyin LuYang LiYanqing MaQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.03496v1 |
|updated| 2024-06-05 17:59:40 UTC |
|summary| Multimodal large language models MLLMs initiated with a trained LLM firstalign images with text and then fine-tune on multimodal mixed inputs. Howeverthe MLLM catastrophically forgets the text-only instructions which do notinclude images and can be addressed within the initial LLM. In this paper wepresent Wings a novel MLLM that excels in both text-only dialogues andmultimodal comprehension. Analyzing MLLM attention in multimodal instructionsreveals that text-only forgetting is related to the attention shifts frompre-image to post-image text. From that we construct extra modules that act asthe boosted learner to compensate for the attention shift. The complementaryvisual and textual learners like wings on either side are connected inparallel within each layers attention block. Initially image and text inputsare aligned with visual learners operating alongside the main attentionbalancing focus on visual elements. Textual learners are later collaborativelyintegrated with attention-based routing to blend the outputs of the visual andtextual learners. We design the Low-Rank Residual Attention LoRRA toguarantee high efficiency for learners. Our experimental results demonstratethat Wings outperforms equally-scaled MLLMs in both text-only and visualquestion-answering tasks. On a newly constructed Interleaved Image-Text IITbenchmark Wings exhibits superior performance from text-only-rich tomultimodal-rich question-answering tasks. |


| Item |Content|
| --- |---|
|idx| 2406.03495v1 |
|title| Grokking Modular Polynomials |
|authors| Darshil DoshiTianyu HeAritra DasAndrey Gromov
|links| http://arxiv.org/abs/2406.03495v1 |
|updated| 2024-06-05 17:59:35 UTC |
|summary| Neural networks readily learn a subset of the modular arithmetic tasks whilefailing to generalize on the rest. This limitation remains unmoved by thechoice of architecture and training strategies. On the other hand ananalytical solution for the weights of Multi-layer Perceptron MLP networksthat generalize on the modular addition task is known in the literature. Inthis work we i extend the class of analytical solutions to include modularmultiplication as well as modular addition with many terms. Additionally weshow that real networks trained on these datasets learn similar solutions upongeneralization grokking. ii We combine these expert solutions toconstruct networks that generalize on arbitrary modular polynomials. iii Wehypothesize a classification of modular polynomials into learnable andnon-learnable via neural networks training and provide experimental evidencesupporting our claims. |


| Item |Content|
| --- |---|
|idx| 2406.03494v1 |
|title| Solving Poisson Equations using Neural Walk-on-Spheres |
|authors| Hong Chul NamJulius BernerAnima Anandkumar
|links| http://arxiv.org/abs/2406.03494v1 |
|updated| 2024-06-05 17:59:22 UTC |
|summary| We propose Neural Walk-on-Spheres NWoS a novel neural PDE solver for theefficient solution of high-dimensional Poisson equations. Leveraging stochasticrepresentations and Walk-on-Spheres methods we develop novel losses for neuralnetworks based on the recursive solution of Poisson equations on spheres insidethe domain. The resulting method is highly parallelizable and does not requirespatial gradients for the loss. We provide a comprehensive comparison againstcompeting methods based on PINNs the Deep Ritz method and backwardstochastic differential equations. In several challenging high-dimensionalnumerical examples we demonstrate the superiority of NWoS in accuracy speedand computational costs. Compared to commonly used PINNs our approach canreduce memory usage and errors by orders of magnitude. Furthermore we applyNWoS to problems in PDE-constrained optimization and molecular dynamics to showits efficiency in practical applications. |


| Item |Content|
| --- |---|
|idx| 2406.03485v1 |
|title| Highway Value Iteration Networks |
|authors| Yuhui WangWeida LiFrancesco FaccioQingyuan WuJürgen Schmidhuber
|links| http://arxiv.org/abs/2406.03485v1 |
|updated| 2024-06-05 17:46:26 UTC |
|summary| Value iteration networks VINs enable end-to-end learning for planning tasksby employing a differentiable planning module that approximates the valueiteration algorithm. However long-term planning remains a challenge becausetraining very deep VINs is difficult. To address this problem we embed highwayvalue iteration -- a recent algorithm designed to facilitate long-term creditassignment -- into the structure of VINs. This improvement augments theplanning module of the VIN with three additional components: 1 an aggregategate which constructs skip connections to improve information flow acrossmany layers 2 an exploration module crafted to increase the diversity ofinformation and gradient flow in spatial dimensions 3 a filter gatedesigned to ensure safe exploration. The resulting novel highway VIN can betrained effectively with hundreds of layers using standard backpropagation. Inlong-term planning tasks requiring hundreds of planning steps deep highwayVINs outperform both traditional VINs and several advanced very deep NNs. |


| Item |Content|
| --- |---|
|idx| 2406.03482v1 |
|title| QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead |
|authors| Amir ZandiehMajid DaliriInsu Han
|links| http://arxiv.org/abs/2406.03482v1 |
|updated| 2024-06-05 17:42:05 UTC |
|summary| Serving LLMs requires substantial memory due to the storage requirements ofKey-Value KV embeddings in the KV cache which grows with sequence length. Aneffective approach to compress KV cache is quantization. However traditionalquantization methods face significant memory overhead due to the need to storequantization constants at least a zero point and a scale in full precisionper data block. Depending on the block size this overhead can add 1 or 2 bitsper quantized number. We introduce QJL a new quantization approach thatconsists of a Johnson-Lindenstrauss JL transform followed by sign-bitquantization. In contrast to existing methods QJL eliminates memory overheadsby removing the need for storing quantization constants. We propose anasymmetric estimator for the inner product of two vectors and demonstrate thatapplying QJL to one vector and a standard JL transform without quantization tothe other provides an unbiased estimator with minimal distortion. We havedeveloped an efficient implementation of the QJL sketch and its correspondinginner product estimator incorporating a lightweight CUDA kernel for optimizedcomputation. When applied across various LLMs and NLP tasks to quantize the KVcache to only 3 bits QJL demonstrates a more than fivefold reduction in KVcache memory usage without compromising accuracy all while achieving fasterruntime. Codes are available at urlhttps://github.com/amirzandieh/QJL. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.03478v1 |
|title| Convolutional Neural Networks and Vision Transformers for Fashion MNIST Classification: A Literature Review |
|authors| Sonia BbouzidiGhazala HciniImen JdeyFadoua Drira
|links| http://arxiv.org/abs/2406.03478v1 |
|updated| 2024-06-05 17:32:22 UTC |
|summary| Our review explores the comparative analysis between Convolutional NeuralNetworks CNNs and Vision Transformers ViTs in the domain of imageclassification with a particular focus on clothing classification within thee-commerce sector. Utilizing the Fashion MNIST dataset we delve into theunique attributes of CNNs and ViTs. While CNNs have long been the cornerstoneof image classification ViTs introduce an innovative self-attention mechanismenabling nuanced weighting of different input data components. Historicallytransformers have primarily been associated with Natural Language ProcessingNLP tasks. Through a comprehensive examination of existing literature ouraim is to unveil the distinctions between ViTs and CNNs in the context of imageclassification. Our analysis meticulously scrutinizes state-of-the-artmethodologies employing both architectures striving to identify the factorsinfluencing their performance. These factors encompass dataset characteristicsimage dimensions the number of target classes hardware infrastructure andthe specific architectures along with their respective top results. Our keygoal is to determine the most appropriate architecture between ViT and CNN forclassifying images in the Fashion MNIST dataset within the e-commerce industrywhile taking into account specific conditions and needs. We highlight theimportance of combining these two architectures with different forms to enhanceoverall performance. By uniting these architectures we can take advantage oftheir unique strengths which may lead to more precise and reliable models fore-commerce applications. CNNs are skilled at recognizing local patterns whileViTs are effective at grasping overall context making their combination apromising strategy for boosting image classification performance. |


| Item |Content|
| --- |---|
|idx| 2406.03474v1 |
|title| AD-H: Autonomous Driving with Hierarchical Agents |
|authors| Zaibin ZhangShiyu TangYuanhang ZhangTalas FuYifan WangYang LiuDong WangJing ShaoLijun WangHuchuan Lu
|links| http://arxiv.org/abs/2406.03474v1 |
|updated| 2024-06-05 17:25:46 UTC |
|summary| Due to the impressive capabilities of multimodal large language modelsMLLMs recent works have focused on employing MLLM-based agents forautonomous driving in large-scale and dynamic environments. However prevalentapproaches often directly translate high-level instructions into low-levelvehicle control signals which deviates from the inherent language generationparadigm of MLLMs and fails to fully harness their emergent powers. As aresult the generalizability of these methods is highly restricted byautonomous driving datasets used during fine-tuning. To tackle this challengewe propose to connect high-level instructions and low-level control signalswith mid-level language-driven commands which are more fine-grained thanhigh-level instructions but more universal and explainable than controlsignals and thus can effectively bridge the gap in between. We implement thisidea through a hierarchical multi-agent driving system named AD-H including aMLLM planner for high-level reasoning and a lightweight controller forlow-level execution. The hierarchical design liberates the MLLM from low-levelcontrol signal decoding and therefore fully releases their emergent capabilityin high-level perception reasoning and planning. We build a new dataset withaction hierarchy annotations. Comprehensive closed-loop evaluations demonstrateseveral key advantages of our proposed AD-H system. First AD-H can notablyoutperform state-of-the-art methods in achieving exceptional drivingperformance even exhibiting self-correction capabilities during vehicleoperation a scenario not encountered in the training dataset. Second AD-Hdemonstrates superior generalization under long-horizon instructions and novelenvironmental conditions significantly surpassing current state-of-the-artmethods. We will make our data and code publicly accessible athttps://github.com/zhangzaibin/AD-H |


| Item |Content|
| --- |---|
|idx| 2406.03461v1 |
|title| Polarization Wavefront Lidar: Learning Large Scene Reconstruction from Polarized Wavefronts |
|authors| Dominik ScheubleChenyang LeiSeung-Hwan BaekMario BijelicFelix Heide
|links| http://arxiv.org/abs/2406.03461v1 |
|updated| 2024-06-05 17:09:51 UTC |
|summary| Lidar has become a cornerstone sensing modality for 3D vision especially forlarge outdoor scenarios and autonomous driving. Conventional lidar sensors arecapable of providing centimeter-accurate distance information by emitting laserpulses into a scene and measuring the time-of-flight ToF of the reflection.However the polarization of the received light that depends on the surfaceorientation and material properties is usually not considered. As such thepolarization modality has the potential to improve scene reconstruction beyonddistance measurements. In this work we introduce a novel long-rangepolarization wavefront lidar sensor PolLidar that modulates the polarizationof the emitted and received light. Departing from conventional lidar sensorsPolLidar allows access to the raw time-resolved polarimetric wavefronts. Weleverage polarimetric wavefronts to estimate normals distance and materialproperties in outdoor scenarios with a novel learned reconstruction method. Totrain and evaluate the method we introduce a simulated and real-worldlong-range dataset with paired raw lidar data ground truth distance andnormal maps. We find that the proposed method improves normal and distancereconstruction by 53 mean angular error and 41 mean absolute error comparedto existing shape-from-polarization SfP and ToF methods. Code and data areopen-sourced at https://light.princeton.edu/pollidar. |


| Item |Content|
| --- |---|
|idx| 2406.03459v1 |
|title| LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection |
|authors| Qiang ChenXiangbo SuXinyu ZhangJian WangJiahui ChenYunpeng ShenChuchu HanZiliang ChenWeixiang XuFanrong LiShan ZhangKun YaoErrui DingGang ZhangJingdong Wang
|links| http://arxiv.org/abs/2406.03459v1 |
|updated| 2024-06-05 17:07:24 UTC |
|summary| In this paper we present a light-weight detection transformer LW-DETRwhich outperforms YOLOs for real-time object detection. The architecture is asimple stack of a ViT encoder a projector and a shallow DETR decoder. Ourapproach leverages recent advanced techniques such as training-effectivetechniques e.g. improved loss and pretraining and interleaved window andglobal attentions for reducing the ViT encoder complexity. We improve the ViTencoder by aggregating multi-level feature maps and the intermediate and finalfeature maps in the ViT encoder forming richer feature maps and introducewindow-major feature map organization for improving the efficiency ofinterleaved attention computation. Experimental results demonstrate that theproposed approach is superior over existing real-time detectors e.g. YOLO andits variants on COCO and other benchmark datasets. Code and models areavailable at https://github.com/Atten4Vis/LW-DETR. |


| Item |Content|
| --- |---|
|idx| 2406.03447v1 |
|title| FILS: Self-Supervised Video Feature Prediction In Semantic Language Space |
|authors| Mona AhmadianFrank GuerinAndrew Gilbert
|links| http://arxiv.org/abs/2406.03447v1 |
|updated| 2024-06-05 16:44:06 UTC |
|summary| This paper demonstrates a self-supervised approach for learning semanticvideo representations. Recent vision studies show that a masking strategy forvision and natural language supervision has contributed to developingtransferable visual pretraining. Our goal is to achieve a more semantic videorepresentation by leveraging the text related to the video content during thepretraining in a fully self-supervised manner. To this end we present FILS anovel self-supervised video Feature prediction In semantic Language SpaceFILS. The vision model can capture valuable structured information bycorrectly predicting masked feature semantics in language space. It is learnedusing a patch-wise video-text contrastive strategy in which the textrepresentations act as prototypes for transforming vision features into alanguage space which are then used as targets for semantically meaningfulfeature prediction using our masked encoder-decoder structure. FILSdemonstrates remarkable transferability on downstream action recognition tasksachieving state-of-the-art on challenging egocentric datasets likeEpic-Kitchens Something-SomethingV2 Charades-Ego and EGTEA using ViT-Base.Our efficient method requires less computation and smaller batches compared toprevious works. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.03495v1 |
|title| Grokking Modular Polynomials |
|authors| Darshil DoshiTianyu HeAritra DasAndrey Gromov
|links| http://arxiv.org/abs/2406.03495v1 |
|updated| 2024-06-05 17:59:35 UTC |
|summary| Neural networks readily learn a subset of the modular arithmetic tasks whilefailing to generalize on the rest. This limitation remains unmoved by thechoice of architecture and training strategies. On the other hand ananalytical solution for the weights of Multi-layer Perceptron MLP networksthat generalize on the modular addition task is known in the literature. Inthis work we i extend the class of analytical solutions to include modularmultiplication as well as modular addition with many terms. Additionally weshow that real networks trained on these datasets learn similar solutions upongeneralization grokking. ii We combine these expert solutions toconstruct networks that generalize on arbitrary modular polynomials. iii Wehypothesize a classification of modular polynomials into learnable andnon-learnable via neural networks training and provide experimental evidencesupporting our claims. |


| Item |Content|
| --- |---|
|idx| 2406.03494v1 |
|title| Solving Poisson Equations using Neural Walk-on-Spheres |
|authors| Hong Chul NamJulius BernerAnima Anandkumar
|links| http://arxiv.org/abs/2406.03494v1 |
|updated| 2024-06-05 17:59:22 UTC |
|summary| We propose Neural Walk-on-Spheres NWoS a novel neural PDE solver for theefficient solution of high-dimensional Poisson equations. Leveraging stochasticrepresentations and Walk-on-Spheres methods we develop novel losses for neuralnetworks based on the recursive solution of Poisson equations on spheres insidethe domain. The resulting method is highly parallelizable and does not requirespatial gradients for the loss. We provide a comprehensive comparison againstcompeting methods based on PINNs the Deep Ritz method and backwardstochastic differential equations. In several challenging high-dimensionalnumerical examples we demonstrate the superiority of NWoS in accuracy speedand computational costs. Compared to commonly used PINNs our approach canreduce memory usage and errors by orders of magnitude. Furthermore we applyNWoS to problems in PDE-constrained optimization and molecular dynamics to showits efficiency in practical applications. |


| Item |Content|
| --- |---|
|idx| 2406.03463v1 |
|title| Gaussian Copula Models for Nonignorable Missing Data Using Auxiliary Marginal Quantiles |
|authors| Joseph FeldmanJerome P. ReiterDaniel R. Kowal
|links| http://arxiv.org/abs/2406.03463v1 |
|updated| 2024-06-05 17:11:59 UTC |
|summary| We present an approach for modeling and imputation of nonignorable missingdata under Gaussian copulas. The analyst posits a set of quantiles of themarginal distributions of the study variables for example reflectinginformation from external data sources or elicited expert opinion. When thesequantiles are accurately specified we prove it is possible to consistentlyestimate the copula correlation and perform multiple imputation in the presenceof nonignorable missing data. We develop algorithms for estimation andimputation that are computationally efficient which we evaluate in simulationstudies of multiple imputation inferences. We apply the model to analyzeassociations between lead exposure levels and end-of-grade test scores for170000 students in North Carolina. These measurements are not missing atrandom as children deemed at-risk for high lead exposure are more likely to bemeasured. We construct plausible marginal quantiles for lead exposure usingnational statistics provided by the Centers for Disease Control and Prevention.Complete cases and missing at random analyses appear to underestimate therelationships between certain variables and end-of-grade test scores whilemultiple imputation inferences under our model support stronger adverseassociations between lead exposure and educational outcomes. |


| Item |Content|
| --- |---|
|idx| 2406.03434v1 |
|title| Unified PAC-Bayesian Study of Pessimism for Offline Policy Learning with Regularized Importance Sampling |
|authors| Imad AoualiVictor-Emmanuel BrunelDavid RohdeAnna Korba
|links| http://arxiv.org/abs/2406.03434v1 |
|updated| 2024-06-05 16:32:14 UTC |
|summary| Off-policy learning OPL often involves minimizing a risk estimator based onimportance weighting to correct bias from the logging policy used to collectdata. However this method can produce an estimator with a high variance. Acommon solution is to regularize the importance weights and learn the policy byminimizing an estimator with penalties derived from generalization boundsspecific to the estimator. This approach known as pessimism has gained recentattention but lacks a unified framework for analysis. To address this gap weintroduce a comprehensive PAC-Bayesian framework to examine pessimism withregularized importance weighting. We derive a tractable PAC-Bayesiangeneralization bound that universally applies to common importance weightregularizations enabling their comparison within a single framework. Ourempirical results challenge common understanding demonstrating theeffectiveness of standard IW regularization techniques. |


| Item |Content|
| --- |---|
|idx| 2406.03396v1 |
|title| Noisy Data Visualization using Functional Data Analysis |
|authors| Haozhe ChenAndres Felipe Duque CorreaGuy WolfKevin R. Moon
|links| http://arxiv.org/abs/2406.03396v1 |
|updated| 2024-06-05 15:53:25 UTC |
|summary| Data visualization via dimensionality reduction is an important tool inexploratory data analysis. However when the data are noisy many existingmethods fail to capture the underlying structure of the data. The method calledEmpirical Intrinsic Geometry EIG was previously proposed for performingdimensionality reduction on high dimensional dynamical processes whiletheoretically eliminating all noise. However implementing EIG in practicerequires the construction of high-dimensional histograms which suffer from thecurse of dimensionality. Here we propose a new data visualization method calledFunctional Information Geometry FIG for dynamical processes that adapts theEIG framework while using approaches from functional data analysis to mitigatethe curse of dimensionality. We experimentally demonstrate that the resultingmethod outperforms a variant of EIG designed for visualization in terms ofcapturing the true structure hyperparameter robustness and computationalspeed. We then use our method to visualize EEG brain measurements of sleepactivity. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.03423v1 |
|title| Improving Users' Passwords with DPAR: a Data-driven Password Recommendation System |
|authors| Assaf MoragLiron DavidEran TochAvishai Wool
|links| http://arxiv.org/abs/2406.03423v1 |
|updated| 2024-06-05 16:19:24 UTC |
|summary| Passwords are the primary authentication method online but even withpassword policies and meters users still find it hard to create strong andmemorable passwords. In this paper we propose DPAR: a Data-driven PAsswordRecommendation system based on a dataset of 905 million leaked passwords. DPARgenerates password recommendations by analyzing the users given password andsuggesting specific tweaks that would make it stronger while still keeping itmemorable and similar to the original password. We conducted two studies toevaluate our approach: verifying the memorability of generated passwordsn317 and evaluating the strength and recall of DPAR recommendations againstpassword meters n441. In a randomized experiment we show that DPARincreased password strength by 34.8 bits on average and did not significantlyaffect the ability to recall their password. Furthermore 36.6 of usersaccepted DPARs recommendations verbatim. We discuss our findings and theirimplications for enhancing password management with recommendation systems. |


| Item |Content|
| --- |---|
|idx| 2406.03415v1 |
|title| RemixTape: Enriching Narratives about Metrics with Semantic Alignment and Contextual Recommendation |
|authors| Matthew BrehmerMargaret DrouhardArjun Srinivasan
|links| http://arxiv.org/abs/2406.03415v1 |
|updated| 2024-06-05 16:11:15 UTC |
|summary| The temporal dynamics of quantitative metrics or key performance indicatorsKPIs are central to decision making within enterprise organizations.Recently major business intelligence providers have introduced newinfrastructure for defining sharing and monitoring metric values. Howeverthese values are often presented in isolation and appropriate context is seldomexternalized. In this design study we present RemixTape an application forconstructing structured narratives around metrics. With design imperativesgrounded in an formative interview study RemixTape provides a hierarchicalcanvas for collecting and coordinating sequences of line chart representationsof metrics along with the ability to externalize situational context aroundthem. RemixTape incorporates affordances to semantically align and annotatejuxtaposed charts and text as well as recommendations of complementary chartsbased on metrics already present on the canvas. We evaluated RemixTape in auser study in which six enterprise data professionals reproduced and extendedpartial narratives with participants appreciating RemixTape as a novelalternative to dashboards galleries and slide presentations for supportingconversations about metrics. We conclude with a reflection on how aspects ofRemixTape could generalize beyond metrics with a call to define a conceptualfoundation for remixing in the context of visualization. |


| Item |Content|
| --- |---|
|idx| 2406.03388v1 |
|title| SelfReDepth: Self-Supervised Real-Time Depth Restoration for Consumer-Grade Sensors |
|authors| Alexandre DuarteFrancisco FernandesJoão M. PereiraCatarina MoreiraJacinto C. NascimentoJoaquim Jorge
|links| http://arxiv.org/abs/2406.03388v1 |
|updated| 2024-06-05 15:38:02 UTC |
|summary| Depth maps produced by consumer-grade sensors suffer from inaccuratemeasurements and missing data from either system or scene-specific sources.Data-driven denoising algorithms can mitigate such problems. However theyrequire vast amounts of ground truth depth data. Recent research has tackledthis limitation using self-supervised learning techniques but it requiresmultiple RGB-D sensors. Moreover most existing approaches focus on denoisingsingle isolated depth maps or specific subjects of interest highlighting aneed for methods to effectively denoise depth maps in real-time dynamicenvironments. This paper extends state-of-the-art approaches fordepth-denoising commodity depth devices proposing SelfReDepth aself-supervised deep learning technique for depth restoration via denoisingand hole-filling by inpainting full-depth maps captured with RGB-D sensors. Thealgorithm targets depth data in video streams utilizing multiple sequentialdepth frames coupled with color data to achieve high-quality depth videos withtemporal coherence. Finally SelfReDepth is designed to be compatible withvarious RGB-D sensors and usable in real-time scenarios as a pre-processingstep before applying other depth-dependent algorithms. Our results demonstrateour approachs real-time performance on real-world datasets. They show that itoutperforms state-of-the-art denoising and restoration performance at over30fps on Commercial Depth Cameras with potential benefits for augmented andmixed-reality applications. |


| Item |Content|
| --- |---|
|idx| 2406.03317v1 |
|title| Save It for the "Hot" Day: An LLM-Empowered Visual Analytics System for Heat Risk Management |
|authors| Haobo LiWong Kam-KwaiYan LuoJuntong ChenChengzhong LiuYaxuan ZhangAlexis Kai Hon LauHuamin QuDongyu Liu
|links| http://arxiv.org/abs/2406.03317v1 |
|updated| 2024-06-05 14:29:44 UTC |
|summary| The escalating frequency and intensity of heat-related climate eventsparticularly heatwaves emphasize the pressing need for advanced heat riskmanagement strategies. Current approaches primarily relying on numericalmodels face challenges in spatial-temporal resolution and in capturing thedynamic interplay of environmental social and behavioral factors affectingheat risks. This has led to difficulties in translating risk assessments intoeffective mitigation actions. Recognizing these problems we introduce a novelapproach leveraging the burgeoning capabilities of Large Language Models LLMsto extract rich and contextual insights from news reports. We hence propose anLLM-empowered visual analytics system Havior that integrates the precisedata-driven insights of numerical models with nuanced news report information.This hybrid approach enables a more comprehensive assessment of heat risks andbetter identification assessment and mitigation of heat-related threats. Thesystem incorporates novel visualization designs such as thermoglyph and newsglyph enhancing intuitive understanding and analysis of heat risks. Theintegration of LLM-based techniques also enables advanced information retrievaland semantic knowledge extraction that can be guided by experts analyticsneeds. Our case studies on two cities that faced significant heatwave eventsand interviews with five experts have demonstrated the usefulness of our systemin providing in-depth and actionable insights for heat risk management. |


| Item |Content|
| --- |---|
|idx| 2406.03245v1 |
|title| Reconfiguring Participatory Design to Resist AI Realism |
|authors| Aakash Gautam
|links| http://dx.doi.org/10.1145/3661455.3669867 |
|updated| 2024-06-05 13:21:46 UTC |
|summary| The growing trend of artificial intelligence AI as a solution to social andtechnical problems reinforces AI Realism -- the belief that AI is an inevitableand natural order. In response this paper argues that participatory designPD with its focus on democratic values and processes can play a role inquestioning and resisting AI Realism. I examine three concerning aspects of AIRealism: the facade of democratization that lacks true empowerment demands forhuman adaptability in contrast to AI systems inflexibility and theobfuscation of essential human labor enabling the AI system. I proposeresisting AI Realism by reconfiguring PD to continue engaging withvalue-centered visions increasing its exploration of non-AI alternatives andmaking the essential human labor underpinning AI systems visible. I position PDas a means to generate friction against AI Realism and open space foralternative futures centered on human needs and values. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.03086v1 |
|title| Task-Oriented Wireless Communications for Collaborative Perception in Intelligent Unmanned Systems |
|authors| Sheng ZhouYukuan JiaRuiqing MaoZhaojun NanYuxuan SunZhisheng Niu
|links| http://arxiv.org/abs/2406.03086v1 |
|updated| 2024-06-05 09:22:19 UTC |
|summary| Collaborative Perception CP has shown great potential to achieve moreholistic and reliable environmental perception in intelligent unmanned systemsIUSs. However implementing CP still faces key challenges due to thecharacteristics of the CP task and the dynamics of wireless channels. In thisarticle a task-oriented wireless communication framework is proposed tojointly optimize the communication scheme and the CP procedure. We firstpropose channel-adaptive compression and robust fusion approaches to extractand exploit the most valuable semantic information under wireless communicationconstraints. We then propose a task-oriented distributed scheduling algorithmto identify the best collaborators for CP under dynamic environments. The mainidea is learning while scheduling where the collaboration utility iseffectively learned with low computation and communication overhead. Casestudies are carried out in connected autonomous driving scenarios to verify theproposed framework. Finally we identify several future research directions. |


| Item |Content|
| --- |---|
|idx| 2406.02890v1 |
|title| Representation Learning For Efficient Deep Multi-Agent Reinforcement Learning |
|authors| Dom HuhPrasant Mohapatra
|links| http://arxiv.org/abs/2406.02890v1 |
|updated| 2024-06-05 03:11:44 UTC |
|summary| Sample efficiency remains a key challenge in multi-agent reinforcementlearning MARL. A promising approach is to learn a meaningful latentrepresentation space through auxiliary learning objectives alongside the MARLobjective to aid in learning a successful control policy. In our work wepresent MAPO-LSO Multi-Agent Policy Optimization with Latent SpaceOptimization which applies a form of comprehensive representation learningdevised to supplement MARL training. Specifically MAPO-LSO proposes amulti-agent extension of transition dynamics reconstruction and self-predictivelearning that constructs a latent state optimization scheme that can betrivially extended to current state-of-the-art MARL algorithms. Empiricalresults demonstrate MAPO-LSO to show notable improvements in sample efficiencyand learning performance compared to its vanilla MARL counterpart without anyadditional MARL hyperparameter tuning on a diverse suite of MARL tasks. |


| Item |Content|
| --- |---|
|idx| 2406.02126v1 |
|title| CityLight: A Universal Model Towards Real-world City-scale Traffic Signal Control Coordination |
|authors| Jinwei ZengChao YuXinyi YangWenxuan AoJian YuanYong LiYu WangHuazhong Yang
|links| http://arxiv.org/abs/2406.02126v1 |
|updated| 2024-06-04 09:10:14 UTC |
|summary| Traffic signal control TSC is a promising low-cost measure to enhancetransportation efficiency without affecting existing road infrastructure. Whilevarious reinforcement learning-based TSC methods have been proposed andexperimentally outperform conventional rule-based methods none of them hasbeen deployed in the real world. An essential gap lies in theoversimplification of the scenarios in terms of intersection heterogeneity androad network intricacy. To make TSC applicable in urban traffic management wetarget TSC coordination in city-scale high-authenticity road networks aimingto solve the three unique and important challenges: city-level scalabilityheterogeneity of real-world intersections and effective coordination amongintricate neighbor connections. Since optimizing multiple agents in aparameter-sharing paradigm can boost the training efficiency and help achievescalability we propose our method CityLight based on the well-acknowledgedoptimization framework parameter-sharing MAPPO. To ensure the unified policynetwork can learn to fit large-scale heterogeneous intersections and tackle theintricate between-neighbor coordination CityLight proposes a universalrepresentation module that consists of two key designs: heterogeneousintersection alignment and neighborhood impact alignment for coordination. Tofurther boost coordination CityLight adopts neighborhood-integrated rewards totransition from achieving local optimal to global optimal. Extensiveexperiments on datasets with hundreds to tens of thousands of real-worldintersections and authentic traffic demands validate the surprisingeffectiveness and generalizability of CityLight with an overall performancegain of 11.66 and a 22.59 improvement in transfer scenarios in terms ofthroughput. |


| Item |Content|
| --- |---|
|idx| 2406.02081v1 |
|title| FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning |
|authors| Wenzhe LiZihan DingSeth KartenChi Jin
|links| http://arxiv.org/abs/2406.02081v1 |
|updated| 2024-06-04 08:04:23 UTC |
|summary| Recent advances in reinforcement learning RL heavily rely on a variety ofwell-designed benchmarks which provide environmental platforms and consistentcriteria to evaluate existing and novel algorithms. Specifically inmulti-agent RL MARL a plethora of benchmarks based on cooperative games havespurred the development of algorithms that improve the scalability ofcooperative multi-agent systems. However for the competitive setting alightweight and open-sourced benchmark with challenging gaming dynamics andvisual inputs has not yet been established. In this work we presentFightLadder a real-time fighting game platform to empower competitive MARLresearch. Along with the platform we provide implementations ofstate-of-the-art MARL algorithms for competitive games as well as a set ofevaluation metrics to characterize the performance and exploitability ofagents. We demonstrate the feasibility of this platform by training a generalagent that consistently defeats 12 built-in characters in single-player modeand expose the difficulty of training a non-exploitable agent without humanknowledge and demonstrations in two-player mode. FightLadder providesmeticulously designed environments to address critical challenges incompetitive MARL research aiming to catalyze a new era of discovery andadvancement in the field. Videos and code athttps://sites.google.com/view/fightladder/home. |


| Item |Content|
| --- |---|
|idx| 2406.02063v1 |
|title| An agent-based model of modal choice with perception biases and habits |
|authors| Carole AdamBenoit Gaudou
|links| http://arxiv.org/abs/2406.02063v1 |
|updated| 2024-06-04 07:44:57 UTC |
|summary| This paper presents an agent-based model of mobility choice influenced byhuman factors such as habits and perception biases. It is implemented in aNetlogo simulator calibrated from results of an online survey aboutperceptions of mobility. The simulator can be played online. It allows tomodify urban infrastructure and observe modal report. |


