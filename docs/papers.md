# cs.CL 

| Item |Content|
| --- |---|
|idx| 2409.10516v1 |
|title| RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval |
|authors| Di LiuMeng ChenBaotong LuHuiqiang JiangZhenhua HanQianxi ZhangQi ChenChengruidong ZhangBailu DingKai ZhangChen ChenFan YangYuqing YangLili Qiu
|links| http://arxiv.org/abs/2409.10516v1 |
|updated| 2024-09-16 17:59:52 UTC |
|summary| Transformer-based large Language Models LLMs become increasingly importantin various domains. However the quadratic time complexity of attentionoperation poses a significant challenge for scaling to longer contexts due tothe extremely high inference latency and GPU memory consumption for cachingkey-value KV vectors. This paper proposes RetrievalAttention a training-freeapproach to accelerate attention computation. To leverage the dynamic sparseproperty of attention RetrievalAttention builds approximate nearest neighborsearch ANNS indexes upon KV vectors in CPU memory and retrieves the mostrelevant ones via vector search during generation. Due to theout-of-distribution OOD between query vectors and key vectors off-the-shelfANNS indexes still need to scan ON usually 30 of all keys data foraccurate retrieval which fails to exploit the high sparsity.RetrievalAttention first identifies the OOD challenge of ANNS-based attentionand addresses it via an attention-aware vector search algorithm that can adaptto queries and only access 1--3 of data thus achieving a sub-linear timecomplexity. RetrievalAttention greatly reduces the inference cost oflong-context LLM with much lower GPU memory requirements while maintaining themodel accuracy. Especially RetrievalAttention only needs 16GB GPU memory forserving 128K tokens in LLMs with 8B parameters which is capable of generatingone token in 0.188 seconds on a single NVIDIA RTX4090 24GB. |


| Item |Content|
| --- |---|
|idx| 2409.10515v1 |
|title| An Efficient Self-Learning Framework For Interactive Spoken Dialog Systems |
|authors| Hitesh TulsianiDavid M. ChanShalini GhoshGarima LalwaniPrabhat PandeyAnkish BansalSri GarimellaAriya RastrowBjörn Hoffmeister
|links| http://arxiv.org/abs/2409.10515v1 |
|updated| 2024-09-16 17:59:50 UTC |
|summary| Dialog systems such as voice assistants are expected to engage with usersin complex evolving conversations. Unfortunately traditional automatic speechrecognition ASR systems deployed in such applications are usually trained torecognize each turn independently and lack the ability to adapt to theconversational context or incorporate user feedback. In this work we introducea general framework for ASR in dialog systems that can go beyond learning fromsingle-turn utterances and learn over time how to adapt to both explicitsupervision and implicit user feedback present in multi-turn conversations. Weaccomplish that by leveraging advances in student-teacher learning andcontext-aware dialog processing and designing contrastive self-supervisionapproaches with Ohm a new online hard-negative mining approach. We show thatleveraging our new framework compared to traditional training leads to relativeWER reductions of close to 10 in real-world dialog systems and up to 26 onpublic synthetic data. |


| Item |Content|
| --- |---|
|idx| 2409.10504v1 |
|title| DILA: Dictionary Label Attention for Mechanistic Interpretability in High-dimensional Multi-label Medical Coding Prediction |
|authors| John WuDavid WuJimeng Sun
|links| http://arxiv.org/abs/2409.10504v1 |
|updated| 2024-09-16 17:45:40 UTC |
|summary| Predicting high-dimensional or extreme multilabels such as in medicalcoding requires both accuracy and interpretability. Existing works often relyon local interpretability methods failing to provide comprehensiveexplanations of the overall mechanism behind each label prediction within amultilabel set. We propose a mechanistic interpretability module calledDIctionary Label Attention method that disentangles uninterpretable denseembeddings into a sparse embedding space where each nonzero element adictionary feature represents a globally learned medical concept. Throughhuman evaluations we show that our sparse embeddings are more humanunderstandable than its dense counterparts by at least 50 percent. Ourautomated dictionary feature identification pipeline leveraging large languagemodels LLMs uncovers thousands of learned medical concepts by examining andsummarizing the highest activating tokens for each dictionary feature. Werepresent the relationships between dictionary features and medical codesthrough a sparse interpretable matrix enhancing the mechanistic and globalunderstanding of the models predictions while maintaining competitiveperformance and scalability without extensive human annotation. |


| Item |Content|
| --- |---|
|idx| 2409.10502v1 |
|title| Causal Language Modeling Can Elicit Search and Reasoning Capabilities on Logic Puzzles |
|authors| Kulin ShahNishanth DikkalaXin WangRina Panigrahy
|links| http://arxiv.org/abs/2409.10502v1 |
|updated| 2024-09-16 17:42:15 UTC |
|summary| Causal language modeling using the Transformer architecture has yieldedremarkable capabilities in Large Language Models LLMs over the last fewyears. However the extent to which fundamental search and reasoningcapabilities emerged within LLMs remains a topic of ongoing debate. In thiswork we study if causal language modeling can learn a complex task such assolving Sudoku puzzles. To solve a Sudoku the model is first required tosearch over all empty cells of the puzzle to decide on a cell to fill and thenapply an appropriate strategy to fill the decided cell. Sometimes theapplication of a strategy only results in thinning down the possible values ina cell rather than concluding the exact value of the cell. In such casesmultiple strategies are applied one after the other to fill a single cell. Weobserve that Transformer models trained on this synthetic task can indeed learnto solve Sudokus our model solves 94.21 of the puzzles fully correctlywhen trained on a logical sequence of steps taken by a solver. We find thattraining Transformers with the logical sequence of steps is necessary andwithout such training they fail to learn Sudoku. We also extend our analysisto Zebra puzzles known as Einstein puzzles and show that the model solves92.04  of the puzzles fully correctly. In addition we study the internalrepresentations of the trained Transformer and find that through linearprobing we can decode information about the set of possible values in anygiven cell from them pointing to the presence of a strong reasoning engineimplicit in the Transformer weights. |


| Item |Content|
| --- |---|
|idx| 2409.10494v1 |
|title| Incorporating Classifier-Free Guidance in Diffusion Model-Based Recommendation |
|authors| Noah BuchananSusan GauchQuan Mai
|links| http://arxiv.org/abs/2409.10494v1 |
|updated| 2024-09-16 17:27:27 UTC |
|summary| This paper presents a diffusion-based recommender system that incorporatesclassifier-free guidance. Most current recommender systems providerecommendations using conventional methods such as collaborative orcontent-based filtering. Diffusion is a new approach to generative AI thatimproves on previous generative AI approaches such as Variational AutoencodersVAEs and Generative Adversarial Networks GANs. We incorporate diffusion ina recommender system that mirrors the sequence users take when browsing andrating items. Although a few current recommender systems incorporate diffusionthey do not incorporate classifier-free guidance a new innovation in diffusionmodels as a whole. In this paper we present a diffusion recommender systemthat augments the underlying recommender system model for improved performanceand also incorporates classifier-free guidance. Our findings show improvementsover state-of-the-art recommender systems for most metrics for severalrecommendation tasks on a variety of datasets. In particular our approachdemonstrates the potential to provide better recommendations when data issparse. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2409.10515v1 |
|title| An Efficient Self-Learning Framework For Interactive Spoken Dialog Systems |
|authors| Hitesh TulsianiDavid M. ChanShalini GhoshGarima LalwaniPrabhat PandeyAnkish BansalSri GarimellaAriya RastrowBjörn Hoffmeister
|links| http://arxiv.org/abs/2409.10515v1 |
|updated| 2024-09-16 17:59:50 UTC |
|summary| Dialog systems such as voice assistants are expected to engage with usersin complex evolving conversations. Unfortunately traditional automatic speechrecognition ASR systems deployed in such applications are usually trained torecognize each turn independently and lack the ability to adapt to theconversational context or incorporate user feedback. In this work we introducea general framework for ASR in dialog systems that can go beyond learning fromsingle-turn utterances and learn over time how to adapt to both explicitsupervision and implicit user feedback present in multi-turn conversations. Weaccomplish that by leveraging advances in student-teacher learning andcontext-aware dialog processing and designing contrastive self-supervisionapproaches with Ohm a new online hard-negative mining approach. We show thatleveraging our new framework compared to traditional training leads to relativeWER reductions of close to 10 in real-world dialog systems and up to 26 onpublic synthetic data. |


| Item |Content|
| --- |---|
|idx| 2409.10496v1 |
|title| MusicLIME: Explainable Multimodal Music Understanding |
|authors| Theodoros SotirouVassilis LyberatosOrfeas Menis MastromichalakisGiorgos Stamou
|links| http://arxiv.org/abs/2409.10496v1 |
|updated| 2024-09-16 17:28:21 UTC |
|summary| Multimodal models are critical for music understanding tasks as they capturethe complex interplay between audio and lyrics. However as these models becomemore prevalent the need for explainability grows-understanding how thesesystems make decisions is vital for ensuring fairness reducing bias andfostering trust. In this paper we introduce MusicLIME a model-agnosticfeature importance explanation method designed for multimodal music models.Unlike traditional unimodal methods which analyze each modality separatelywithout considering the interaction between them often leading to incompleteor misleading explanations MusicLIME reveals how audio and lyrical featuresinteract and contribute to predictions providing a holistic view of themodels decision-making. Additionally we enhance local explanations byaggregating them into global explanations giving users a broader perspectiveof model behavior. Through this work we contribute to improving theinterpretability of multimodal music models empowering users to make informedchoices and fostering more equitable fair and transparent musicunderstanding systems. |


| Item |Content|
| --- |---|
|idx| 2409.10489v2 |
|title| Flash STU: Fast Spectral Transform Units |
|authors| Y. Isabel LiuWindsor NguyenYagiz DevreEvan DogariuAnirudha MajumdarElad Hazan
|links| http://arxiv.org/abs/2409.10489v2 |
|updated| 2024-09-17 12:01:14 UTC |
|summary| This paper describes an efficient open source PyTorch implementation of theSpectral Transform Unit. We investigate sequence prediction tasks over severalmodalities including language robotics and simulated dynamical systems. Wefind that for the same parameter count the STU and its variants outperform theTransformer as well as other leading state space models across variousmodalities. |


| Item |Content|
| --- |---|
|idx| 2409.10488v1 |
|title| Do Pre-trained Vision-Language Models Encode Object States? |
|authors| Kaleb NewmanShijie WangYuan ZangDavid HeffrenChen Sun
|links| http://arxiv.org/abs/2409.10488v1 |
|updated| 2024-09-16 17:22:18 UTC |
|summary| For a vision-language model VLM to understand the physical world such ascause and effect a first step is to capture the temporal dynamics of thevisual world for example how the physical states of objects evolve over timee.g. a whole apple into a sliced apple. Our paper aims to investigate if VLMspre-trained on web-scale data learn to encode object states which can beextracted with zero-shot text prompts. We curate an object state recognitiondataset ChangeIt-Frames and evaluate nine open-source VLMs including modelstrained with contrastive and generative objectives. We observe that while thesestate-of-the-art vision-language models can reliably perform objectrecognition they consistently fail to accurately distinguish the objectsphysical states. Through extensive experiments we identify three areas forimprovements for VLMs to better encode object states namely the quality ofobject localization the architecture to bind concepts to objects and theobjective to learn discriminative visual and language encoders on objectstates. Data and code are released. |


| Item |Content|
| --- |---|
|idx| 2409.10481v1 |
|title| Exploring 3D Face Reconstruction and Fusion Methods for Face Verification: A Case-Study in Video Surveillance |
|authors| Simone Maurizio La CavaSara ConcasRuben TolosanaRoberto CasulaGiulia OrrùMartin DrahanskyJulian FierrezGian Luca Marcialis
|links| http://arxiv.org/abs/2409.10481v1 |
|updated| 2024-09-16 17:17:47 UTC |
|summary| 3D face reconstruction 3DFR algorithms are based on specific assumptionstailored to distinct application scenarios. These assumptions limit their usewhen acquisition conditions such as the subjects distance from the camera orthe cameras characteristics are different than expected as typically happensin video surveillance. Additionally 3DFR algorithms follow various strategiesto address the reconstruction of a 3D shape from 2D data such as statisticalmodel fitting photometric stereo or deep learning. In the present study weexplore the application of three 3DFR algorithms representative of the SOTAemploying each one as the template set generator for a face verificationsystem. The scores provided by each system are combined by score-level fusion.We show that the complementarity induced by different 3DFR algorithms improvesperformance when tests are conducted at never-seen-before distances from thecamera and camera characteristics cross-distance and cross-camera settingsthus encouraging further investigations on multiple 3DFR-based approaches. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2409.10516v1 |
|title| RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval |
|authors| Di LiuMeng ChenBaotong LuHuiqiang JiangZhenhua HanQianxi ZhangQi ChenChengruidong ZhangBailu DingKai ZhangChen ChenFan YangYuqing YangLili Qiu
|links| http://arxiv.org/abs/2409.10516v1 |
|updated| 2024-09-16 17:59:52 UTC |
|summary| Transformer-based large Language Models LLMs become increasingly importantin various domains. However the quadratic time complexity of attentionoperation poses a significant challenge for scaling to longer contexts due tothe extremely high inference latency and GPU memory consumption for cachingkey-value KV vectors. This paper proposes RetrievalAttention a training-freeapproach to accelerate attention computation. To leverage the dynamic sparseproperty of attention RetrievalAttention builds approximate nearest neighborsearch ANNS indexes upon KV vectors in CPU memory and retrieves the mostrelevant ones via vector search during generation. Due to theout-of-distribution OOD between query vectors and key vectors off-the-shelfANNS indexes still need to scan ON usually 30 of all keys data foraccurate retrieval which fails to exploit the high sparsity.RetrievalAttention first identifies the OOD challenge of ANNS-based attentionand addresses it via an attention-aware vector search algorithm that can adaptto queries and only access 1--3 of data thus achieving a sub-linear timecomplexity. RetrievalAttention greatly reduces the inference cost oflong-context LLM with much lower GPU memory requirements while maintaining themodel accuracy. Especially RetrievalAttention only needs 16GB GPU memory forserving 128K tokens in LLMs with 8B parameters which is capable of generatingone token in 0.188 seconds on a single NVIDIA RTX4090 24GB. |


| Item |Content|
| --- |---|
|idx| 2409.10502v1 |
|title| Causal Language Modeling Can Elicit Search and Reasoning Capabilities on Logic Puzzles |
|authors| Kulin ShahNishanth DikkalaXin WangRina Panigrahy
|links| http://arxiv.org/abs/2409.10502v1 |
|updated| 2024-09-16 17:42:15 UTC |
|summary| Causal language modeling using the Transformer architecture has yieldedremarkable capabilities in Large Language Models LLMs over the last fewyears. However the extent to which fundamental search and reasoningcapabilities emerged within LLMs remains a topic of ongoing debate. In thiswork we study if causal language modeling can learn a complex task such assolving Sudoku puzzles. To solve a Sudoku the model is first required tosearch over all empty cells of the puzzle to decide on a cell to fill and thenapply an appropriate strategy to fill the decided cell. Sometimes theapplication of a strategy only results in thinning down the possible values ina cell rather than concluding the exact value of the cell. In such casesmultiple strategies are applied one after the other to fill a single cell. Weobserve that Transformer models trained on this synthetic task can indeed learnto solve Sudokus our model solves 94.21 of the puzzles fully correctlywhen trained on a logical sequence of steps taken by a solver. We find thattraining Transformers with the logical sequence of steps is necessary andwithout such training they fail to learn Sudoku. We also extend our analysisto Zebra puzzles known as Einstein puzzles and show that the model solves92.04  of the puzzles fully correctly. In addition we study the internalrepresentations of the trained Transformer and find that through linearprobing we can decode information about the set of possible values in anygiven cell from them pointing to the presence of a strong reasoning engineimplicit in the Transformer weights. |


| Item |Content|
| --- |---|
|idx| 2409.10499v1 |
|title| Partial Distribution Matching via Partial Wasserstein Adversarial Networks |
|authors| Zi-Ming WangNan XueLing LeiRebecka JörnstenGui-Song Xia
|links| http://arxiv.org/abs/2409.10499v1 |
|updated| 2024-09-16 17:41:45 UTC |
|summary| This paper studies the problem of distribution matching DM which is afundamental machine learning problem seeking to robustly align two probabilitydistributions. Our approach is established on a relaxed formulation calledpartial distribution matching PDM which seeks to match a fraction of thedistributions instead of matching them completely. We theoretically derive theKantorovich-Rubinstein duality for the partial Wasserstain-1 PW discrepancyand develop a partial Wasserstein adversarial network PWAN that efficientlyapproximates the PW discrepancy based on this dual form. Partial matching canthen be achieved by optimizing the network using gradient descent. Twopractical tasks point set registration and partial domain adaptation areinvestigated where the goals are to partially match distributions in 3D spaceand high-dimensional feature space respectively. The experiment results confirmthat the proposed PWAN effectively produces highly robust matching resultsperforming better or on par with the state-of-the-art methods. |


| Item |Content|
| --- |---|
|idx| 2409.10496v1 |
|title| MusicLIME: Explainable Multimodal Music Understanding |
|authors| Theodoros SotirouVassilis LyberatosOrfeas Menis MastromichalakisGiorgos Stamou
|links| http://arxiv.org/abs/2409.10496v1 |
|updated| 2024-09-16 17:28:21 UTC |
|summary| Multimodal models are critical for music understanding tasks as they capturethe complex interplay between audio and lyrics. However as these models becomemore prevalent the need for explainability grows-understanding how thesesystems make decisions is vital for ensuring fairness reducing bias andfostering trust. In this paper we introduce MusicLIME a model-agnosticfeature importance explanation method designed for multimodal music models.Unlike traditional unimodal methods which analyze each modality separatelywithout considering the interaction between them often leading to incompleteor misleading explanations MusicLIME reveals how audio and lyrical featuresinteract and contribute to predictions providing a holistic view of themodels decision-making. Additionally we enhance local explanations byaggregating them into global explanations giving users a broader perspectiveof model behavior. Through this work we contribute to improving theinterpretability of multimodal music models empowering users to make informedchoices and fostering more equitable fair and transparent musicunderstanding systems. |


| Item |Content|
| --- |---|
|idx| 2409.10489v2 |
|title| Flash STU: Fast Spectral Transform Units |
|authors| Y. Isabel LiuWindsor NguyenYagiz DevreEvan DogariuAnirudha MajumdarElad Hazan
|links| http://arxiv.org/abs/2409.10489v2 |
|updated| 2024-09-17 12:01:14 UTC |
|summary| This paper describes an efficient open source PyTorch implementation of theSpectral Transform Unit. We investigate sequence prediction tasks over severalmodalities including language robotics and simulated dynamical systems. Wefind that for the same parameter count the STU and its variants outperform theTransformer as well as other leading state space models across variousmodalities. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2409.10488v1 |
|title| Do Pre-trained Vision-Language Models Encode Object States? |
|authors| Kaleb NewmanShijie WangYuan ZangDavid HeffrenChen Sun
|links| http://arxiv.org/abs/2409.10488v1 |
|updated| 2024-09-16 17:22:18 UTC |
|summary| For a vision-language model VLM to understand the physical world such ascause and effect a first step is to capture the temporal dynamics of thevisual world for example how the physical states of objects evolve over timee.g. a whole apple into a sliced apple. Our paper aims to investigate if VLMspre-trained on web-scale data learn to encode object states which can beextracted with zero-shot text prompts. We curate an object state recognitiondataset ChangeIt-Frames and evaluate nine open-source VLMs including modelstrained with contrastive and generative objectives. We observe that while thesestate-of-the-art vision-language models can reliably perform objectrecognition they consistently fail to accurately distinguish the objectsphysical states. Through extensive experiments we identify three areas forimprovements for VLMs to better encode object states namely the quality ofobject localization the architecture to bind concepts to objects and theobjective to learn discriminative visual and language encoders on objectstates. Data and code are released. |


| Item |Content|
| --- |---|
|idx| 2409.10481v1 |
|title| Exploring 3D Face Reconstruction and Fusion Methods for Face Verification: A Case-Study in Video Surveillance |
|authors| Simone Maurizio La CavaSara ConcasRuben TolosanaRoberto CasulaGiulia OrrùMartin DrahanskyJulian FierrezGian Luca Marcialis
|links| http://arxiv.org/abs/2409.10481v1 |
|updated| 2024-09-16 17:17:47 UTC |
|summary| 3D face reconstruction 3DFR algorithms are based on specific assumptionstailored to distinct application scenarios. These assumptions limit their usewhen acquisition conditions such as the subjects distance from the camera orthe cameras characteristics are different than expected as typically happensin video surveillance. Additionally 3DFR algorithms follow various strategiesto address the reconstruction of a 3D shape from 2D data such as statisticalmodel fitting photometric stereo or deep learning. In the present study weexplore the application of three 3DFR algorithms representative of the SOTAemploying each one as the template set generator for a face verificationsystem. The scores provided by each system are combined by score-level fusion.We show that the complementarity induced by different 3DFR algorithms improvesperformance when tests are conducted at never-seen-before distances from thecamera and camera characteristics cross-distance and cross-camera settingsthus encouraging further investigations on multiple 3DFR-based approaches. |


| Item |Content|
| --- |---|
|idx| 2409.10476v1 |
|title| SimInversion: A Simple Framework for Inversion-Based Text-to-Image Editing |
|authors| Qi QianHaiyang XuMing YanJuhua Hu
|links| http://arxiv.org/abs/2409.10476v1 |
|updated| 2024-09-16 17:10:50 UTC |
|summary| Diffusion models demonstrate impressive image generation performance withtext guidance. Inspired by the learning process of diffusion existing imagescan be edited according to text by DDIM inversion. However the vanilla DDIMinversion is not optimized for classifier-free guidance and the accumulatederror will result in the undesired performance. While many algorithms aredeveloped to improve the framework of DDIM inversion for editing in this workwe investigate the approximation error in DDIM inversion and propose todisentangle the guidance scale for the source and target branches to reduce theerror while keeping the original framework. Moreover a better guidance scalei.e. 0.5 than default settings can be derived theoretically. Experiments onPIE-Bench show that our proposal can improve the performance of DDIM inversiondramatically without sacrificing efficiency. |


| Item |Content|
| --- |---|
|idx| 2409.10473v1 |
|title| MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion |
|authors| Lehong WuLilang LinJiahang ZhangYiyang MaJiaying Liu
|links| http://arxiv.org/abs/2409.10473v1 |
|updated| 2024-09-16 17:06:10 UTC |
|summary| Self-supervised learning has proved effective for skeleton-based human actionunderstanding. However previous works either rely on contrastive learning thatsuffers false negative problems or are based on reconstruction that learns toomuch unessential low-level clues leading to limited representations fordownstream tasks. Recently great advances have been made in generativelearning which is naturally a challenging yet meaningful pretext task to modelthe general underlying data distributions. However the representation learningcapacity of generative models is under-explored especially for the skeletonswith spacial sparsity and temporal redundancy. To this end we propose MaskedConditional Diffusion MacDiff as a unified framework for human skeletonmodeling. For the first time we leverage diffusion models as effectiveskeleton representation learners. Specifically we train a diffusion decoderconditioned on the representations extracted by a semantic encoder. Randommasking is applied to encoder inputs to introduce a information bottleneck andremove redundancy of skeletons. Furthermore we theoretically demonstrate thatour generative objective involves the contrastive learning objective whichaligns the masked and noisy views. Meanwhile it also enforces therepresentation to complement for the noisy view leading to bettergeneralization performance. MacDiff achieves state-of-the-art performance onrepresentation learning benchmarks while maintaining the competence forgenerative tasks. Moreover we leverage the diffusion model for dataaugmentation significantly enhancing the fine-tuning performance in scenarioswith scarce labeled data. Our project is available athttps://lehongwu.github.io/ECCV24MacDiff/. |


| Item |Content|
| --- |---|
|idx| 2409.10445v1 |
|title| Deep-Wide Learning Assistance for Insect Pest Classification |
|authors| Toan NguyenHuy NguyenHuy UngHieu UngBinh Nguyen
|links| http://arxiv.org/abs/2409.10445v1 |
|updated| 2024-09-16 16:29:41 UTC |
|summary| Accurate insect pest recognition plays a critical role in agriculture. It isa challenging problem due to the intricate characteristics of insects. In thispaper we present DeWi novel learning assistance for insect pestclassification. With a one-stage and alternating training strategy DeWisimultaneously improves several Convolutional Neural Networks in twoperspectives: discrimination by optimizing a triplet margin loss in asupervised training manner and generalization via data augmentation. Fromthat DeWi can learn discriminative and in-depth features of insect pestsdeep yet still generalize well to a large number of insect categories wide.Experimental results show that DeWi achieves the highest performances on twoinsect pest classification benchmarks 76.44 accuracy on the IP102 datasetand 99.79 accuracy on the D0 dataset respectively. In addition extensiveevaluations and ablation studies are conducted to thoroughly investigate ourDeWi and demonstrate its superiority. Our source code is available athttps://github.com/toannguyen1904/DeWi. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2409.10499v1 |
|title| Partial Distribution Matching via Partial Wasserstein Adversarial Networks |
|authors| Zi-Ming WangNan XueLing LeiRebecka JörnstenGui-Song Xia
|links| http://arxiv.org/abs/2409.10499v1 |
|updated| 2024-09-16 17:41:45 UTC |
|summary| This paper studies the problem of distribution matching DM which is afundamental machine learning problem seeking to robustly align two probabilitydistributions. Our approach is established on a relaxed formulation calledpartial distribution matching PDM which seeks to match a fraction of thedistributions instead of matching them completely. We theoretically derive theKantorovich-Rubinstein duality for the partial Wasserstain-1 PW discrepancyand develop a partial Wasserstein adversarial network PWAN that efficientlyapproximates the PW discrepancy based on this dual form. Partial matching canthen be achieved by optimizing the network using gradient descent. Twopractical tasks point set registration and partial domain adaptation areinvestigated where the goals are to partially match distributions in 3D spaceand high-dimensional feature space respectively. The experiment results confirmthat the proposed PWAN effectively produces highly robust matching resultsperforming better or on par with the state-of-the-art methods. |


| Item |Content|
| --- |---|
|idx| 2409.10463v1 |
|title| Kolmogorov-Arnold Networks in Low-Data Regimes: A Comparative Study with Multilayer Perceptrons |
|authors| Farhad Pourkamali-Anaraki
|links| http://arxiv.org/abs/2409.10463v1 |
|updated| 2024-09-16 16:56:08 UTC |
|summary| Multilayer Perceptrons MLPs have long been a cornerstone in deep learningknown for their capacity to model complex relationships. RecentlyKolmogorov-Arnold Networks KANs have emerged as a compelling alternativeutilizing highly flexible learnable activation functions directly on networkedges a departure from the neuron-centric approach of MLPs. However KANssignificantly increase the number of learnable parameters raising concernsabout their effectiveness in data-scarce environments. This paper presents acomprehensive comparative study of MLPs and KANs from both algorithmic andexperimental perspectives with a focus on low-data regimes. We introduce aneffective technique for designing MLPs with unique parameterized activationfunctions for each neuron enabling a more balanced comparison with KANs. Usingempirical evaluations on simulated data and two real-world data sets frommedicine and engineering we explore the trade-offs between model complexityand accuracy with particular attention to the role of network depth. Ourfindings show that MLPs with individualized activation functions achievesignificantly higher predictive accuracy with only a modest increase inparameters especially when the sample size is limited to around one hundred.For example in a three-class classification problem within additivemanufacturing MLPs achieve a median accuracy of 0.91 significantlyoutperforming KANs which only reach a median accuracy of 0.53 with defaulthyperparameters. These results offer valuable insights into the impact ofactivation function selection in neural networks. |


| Item |Content|
| --- |---|
|idx| 2409.10421v1 |
|title| Multidimensional Deconvolution with Profiling |
|authors| Huanbiao ZhuKrish DesaiMikael KuuselaVinicius MikuniBenjamin NachmanLarry Wasserman
|links| http://arxiv.org/abs/2409.10421v1 |
|updated| 2024-09-16 15:52:28 UTC |
|summary| In many experimental contexts it is necessary to statistically remove theimpact of instrumental effects in order to physically interpret measurements.This task has been extensively studied in particle physics where thedeconvolution task is called unfolding. A number of recent methods have shownhow to perform high-dimensional unbinned unfolding using machine learning.However one of the assumptions in all of these methods is that the detectorresponse is accurately modeled in the Monte Carlo simulation. In practice thedetector response depends on a number of nuisance parameters that can beconstrained with data. We propose a new algorithm called Profile OmniFoldPOF which works in a similar iterative manner as the OmniFold OF algorithmwhile being able to simultaneously profile the nuisance parameters. Weillustrate the method with a Gaussian example as a proof of concepthighlighting its promising capabilities. |


| Item |Content|
| --- |---|
|idx| 2409.10139v1 |
|title| Towards Explainable Automated Data Quality Enhancement without Domain Knowledge |
|authors| Djibril Sarr
|links| http://arxiv.org/abs/2409.10139v1 |
|updated| 2024-09-16 10:08:05 UTC |
|summary| In the era of big data ensuring the quality of datasets has becomeincreasingly crucial across various domains. We propose a comprehensiveframework designed to automatically assess and rectify data quality issues inany given dataset regardless of its specific content focusing on both textualand numerical data. Our primary objective is to address three fundamental typesof defects: absence redundancy and incoherence. At the heart of our approachlies a rigorous demand for both explainability and interpretability ensuringthat the rationale behind the identification and correction of data anomaliesis transparent and understandable. To achieve this we adopt a hybrid approachthat integrates statistical methods with machine learning algorithms. Indeedby leveraging statistical techniques alongside machine learning we strike abalance between accuracy and explainability enabling users to trust andcomprehend the assessment process. Acknowledging the challenges associated withautomating the data quality assessment process particularly in terms of timeefficiency and accuracy we adopt a pragmatic strategy employingresource-intensive algorithms only when necessary while favoring simpler moreefficient solutions whenever possible. Through a practical analysis conductedon a publicly provided dataset we illustrate the challenges that arise whentrying to enhance data quality while keeping explainability. We demonstrate theeffectiveness of our approach in detecting and rectifying missing valuesduplicates and typographical errors as well as the challenges remaining to beaddressed to achieve similar accuracy on statistical outliers and logic errorsunder the constraints set in our work. |


| Item |Content|
| --- |---|
|idx| 2409.10096v1 |
|title| Robust Reinforcement Learning with Dynamic Distortion Risk Measures |
|authors| Anthony CoacheSebastian Jaimungal
|links| http://arxiv.org/abs/2409.10096v1 |
|updated| 2024-09-16 08:54:59 UTC |
|summary| In a reinforcement learning RL setting the agents optimal strategyheavily depends on her risk preferences and the underlying model dynamics ofthe training environment. These two aspects influence the agents ability tomake well-informed and time-consistent decisions when facing testingenvironments. In this work we devise a framework to solve robust risk-aware RLproblems where we simultaneously account for environmental uncertainty and riskwith a class of dynamic robust distortion risk measures. Robustness isintroduced by considering all models within a Wasserstein ball around areference model. We estimate such dynamic robust risk measures using neuralnetworks by making use of strictly consistent scoring functions derive policygradient formulae using the quantile representation of distortion riskmeasures and construct an actor-critic algorithm to solve this class of robustrisk-aware RL problems. We demonstrate the performance of our algorithm on aportfolio allocation example. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2409.10459v1 |
|title| Efficiently Crowdsourcing Visual Importance with Punch-Hole Annotation |
|authors| Minsuk ChangSoohyun LeeAeri ChoHyeon JeonSeokhyeon ParkCindy Xiong BearfieldJinwook Seo
|links| http://arxiv.org/abs/2409.10459v1 |
|updated| 2024-09-16 16:49:59 UTC |
|summary| We introduce a novel crowdsourcing method for identifying important areas ingraphical images through punch-hole labeling. Traditional methods such as gazetrackers and mouse-based annotations which generate continuous data can beimpractical in crowdsourcing scenarios. They require many participants and theoutcome data can be noisy. In contrast our method first segments the graphicalimage with a grid and drops a portion of the patches punch holes. Then weiteratively ask the labeler to validate each annotation with holes narrowingdown the annotation only having the most important area. This approach aims toreduce annotation noise in crowdsourcing by standardizing the annotations whileenhancing labeling efficiency and reliability. Preliminary findings fromfundamental charts demonstrate that punch-hole labeling can effectivelypinpoint critical regions. This also highlights its potential for broaderapplication in visualization research particularly in studying large-scaleusers graphical perception. Our future work aims to enhance the algorithm toachieve faster labeling speed and prove its utility through large-scaleexperiments. |


| Item |Content|
| --- |---|
|idx| 2409.10450v1 |
|title| Charting EDA: Characterizing Interactive Visualization Use in Computational Notebooks with a Mixed-Methods Formalism |
|authors| Dylan WoottonAmy Rae FoxEvan PeckArvind Satyanarayan
|links| http://arxiv.org/abs/2409.10450v1 |
|updated| 2024-09-16 16:37:21 UTC |
|summary| Interactive visualizations are powerful tools for Exploratory Data AnalysisEDA but how do they affect the observations analysts make about their dataWe conducted a qualitative experiment with 13 professional data scientistsanalyzing two datasets with Jupyter notebooks collecting a rich dataset ofinteraction traces and think-aloud utterances. By qualitatively codingparticipant utterances we introduce a formalism that describes EDA as asequence of analysis states where each state is comprised of either arepresentation an analyst constructs e.g. the output of a data frame aninteractive visualization etc. or an observation the analyst makes e.g.about missing data the relationship between variables etc.. By applying ourformalism to our dataset we identify that interactive visualizations onaverage lead to earlier and more complex insights about relationships betweendataset attributes compared to static visualizations. Moreover by calculatingmetrics such as revisit count and representational diversity we uncover thatsome representations serve more as planning aids during EDA rather than toolsstrictly for hypothesis-answering. We show how these measures help identifyother patterns of analysis behavior such as the 80-20 rule where a smallsubset of representations drove the majority of observations. Based on thesefindings we offer design guidelines for interactive exploratory analysistooling and reflect on future directions for studying the role thatvisualizations play in EDA. |


| Item |Content|
| --- |---|
|idx| 2409.10446v1 |
|title| KoroT-3E: A Personalized Musical Mnemonics Tool for Enhancing Memory Retention of Complex Computer Science Concepts |
|authors| Xiangzhe YuanJiajun WangSiying HuAndrew CheungZhicong Lu
|links| http://arxiv.org/abs/2409.10446v1 |
|updated| 2024-09-16 16:35:50 UTC |
|summary| As the demand for computer science CS skills grows mastering foundationalconcepts is crucial yet challenging for novice learners. To address thischallenge we present KoroT-3E an AI-based system that creates personalizedmusical mnemonics to enhance both memory retention and understanding ofconcepts in CS. KoroT-3E enables users to transform complex concepts intomemorable lyrics and compose melodies that suit their musical preferences. Weconducted semi-structured interviews n12 to investigate why novice learnersfind it challenging to memorize and understand CS concepts. The findingscombined with constructivist learning theory established our initial designwhich was then refined following consultations with CS education experts. Anempirical experimentn36 showed that those using KoroT-3E n18significantly outperformed the control group n18 with improved memoryefficiency increased motivation and a positive learning experience. Thesefindings demonstrate the effectiveness of integrating multimodal generative AIinto CS education to create personalized and interactive learning experiences. |


| Item |Content|
| --- |---|
|idx| 2409.10354v2 |
|title| Learnings from a Large-Scale Deployment of an LLM-Powered Expert-in-the-Loop Healthcare Chatbot |
|authors| Bhuvan SachdevaPragnya RamjeeGeeta FulariKaushik MuraliMohit Jain
|links| http://arxiv.org/abs/2409.10354v2 |
|updated| 2024-09-17 03:22:45 UTC |
|summary| Large Language Models LLMs are widely used in healthcare but limitationslike hallucinations incomplete information and bias hinder their reliability.To address these researchers released the Build Your Own expert Bot BYOeBplatform enabling developers to create LLM-powered chatbots with integratedexpert verification. CataractBot its first implementation providesexpert-verified responses to cataract surgery questions. A pilot evaluationshowed its potential however the study had a small sample size and wasprimarily qualitative. In this work we conducted a large-scale 24-weekdeployment of CataractBot involving 318 patients and attendants who sent 1992messages with 91.71 of responses verified by seven experts. Analysis ofinteraction logs revealed that medical questions significantly outnumberedlogistical ones hallucinations were negligible and experts rated 84.52 ofmedical answers as accurate. As the knowledge base expanded with expertcorrections system performance improved by 19.02 reducing expert workload.These insights guide the design of future LLM-powered chatbots. |


| Item |Content|
| --- |---|
|idx| 2409.10258v1 |
|title| Co-Designing Dynamic Mixed Reality Drill Positioning Widgets: A Collaborative Approach with Dentists in a Realistic Setup |
|authors| Mine DastanMichele FiorentinoElias D. WalterChristian DiegritzAntonio E. UvaUlrich EckNassir Navab
|links| http://dx.doi.org/10.1109/TVCG.2024.3456149 |
|updated| 2024-09-16 13:10:37 UTC |
|summary| Mixed Reality MR is proven in the literature to support precise spatialdental drill positioning by superimposing 3D widgets. Despite this the relatedknowledge about widgets visual design and interactive user feedback is stilllimited. Therefore this study is contributed to by co-designed MR drill toolpositioning widgets with two expert dentists and three MR experts. The resultsof co-design are two static widgets SWs: a simple entry point a target axisand two dynamic widgets DWs variants of dynamic error visualization with andwithout a target axis DWTA and DWEP. We evaluated the co-designed widgets ina virtual reality simulation supported by a realistic setup with a trackedphantom patient a virtual magnifying loupe and a dentists foot pedal. Theuser study involved 35 dentists with various backgrounds and years ofexperience. The findings demonstrated significant results DWs outperform SWsin positional and rotational precision especially with younger generations andsubjects with gaming experiences. The user preference remains for DWs 19instead of SWs 16. However findings indicated that the precision positivelycorrelates with the time trade-off. The post-experience questionnaireNASA-TLX showed that DWs increase mental and physical demand effort andfrustration more than SWs. Comparisons between DWEP and DWTA show that the DWscomplexity level influences time physical and mental demands. The DWs areextensible to diverse medical and industrial scenarios that demand precision. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2409.10395v1 |
|title| Reducing Leximin Fairness to Utilitarian Optimization |
|authors| Eden HartmanYonatan AumannAvinatan HassidimErel Segal-Halevi
|links| http://arxiv.org/abs/2409.10395v1 |
|updated| 2024-09-16 15:31:41 UTC |
|summary| Two prominent objectives in social choice are utilitarian - maximizing thesum of agents utilities and leximin - maximizing the smallest agentsutility then the second-smallest etc. Utilitarianism is typicallycomputationally easier to attain but is generally viewed as less fair. Thispaper presents a general reduction scheme that given a utilitarian solverproduces a distribution over outcomes that is leximin in expectation.Importantly the scheme is robust in the sense that given an approximateutilitarian solver it produces an outcome that is approximately-leximin inexpectation - with the same approximation factor. We apply our scheme toseveral social choice problems: stochastic allocations of indivisible goodsgiveaway lotteries and fair lotteries for participatory budgeting. |


| Item |Content|
| --- |---|
|idx| 2409.10215v1 |
|title| Synchronization-Based Cooperative Distributed Model Predictive Control |
|authors| Julius BeerwerthMaximilian KloockBassam Alrifaee
|links| http://arxiv.org/abs/2409.10215v1 |
|updated| 2024-09-16 12:06:06 UTC |
|summary| Distributed control algorithms are known to reduce overall computation timecompared to centralized control algorithms. However they can result ininconsistent solutions leading to the violation of safety-critical constraints.Inconsistent solutions can arise when two or more agents compute concurrentlywhile making predictions on each others control actions. To address this issuewe propose an iterative algorithm called Synchronization-Based CooperativeDistributed Model Predictive Control which we presented in 1. The algorithmconsists of two steps: 1. computing the optimal control inputs for each agentand 2. synchronizing the predicted states across all agents. We demonstrate theefficacy of our algorithm in the control of multiple small-scale vehicles inour Cyber-Physical Mobility Lab. |


| Item |Content|
| --- |---|
|idx| 2409.10117v1 |
|title| Multi-Agent Obstacle Avoidance using Velocity Obstacles and Control Barrier Functions |
|authors| Alejandro Sánchez RonceroRafael I. Cabral MuchachoPetter Ögren
|links| http://arxiv.org/abs/2409.10117v1 |
|updated| 2024-09-16 09:29:38 UTC |
|summary| Velocity Obstacles VO methods form a paradigm for collision avoidancestrategies among moving obstacles and agents. While VO methods perform well insimple multi-agent environments they dont guarantee safety and can showoverly conservative behavior in common situations. In this paper we propose tocombine a VO-strategy for guidance with a CBF-approach for safety whichovercomes the overly conservative behavior of VOs and formally guaranteessafety. We validate our method in a baseline comparison study using 2nd orderintegrator and car-like dynamics. Results support that our method outperformsthe baselines w.r.t. path smoothness collision avoidance and success rates. |


| Item |Content|
| --- |---|
|idx| 2409.10047v1 |
|title| Bearing-Distance Based Flocking with Zone-Based Interactions |
|authors| Hossein B. Jond
|links| http://arxiv.org/abs/2409.10047v1 |
|updated| 2024-09-16 07:20:29 UTC |
|summary| This paper presents a novel zone-based flocking control approach suitable fordynamic multi-agent systems MAS. Inspired by Reynolds behavioral rules forboids flocking behavioral rules with the zones of repulsion conflictattraction and surveillance are introduced. For each agent using only bearingand distance measurements behavioral deviation vectors quantify the deviationsfrom the local separation local and global flock velocity alignment localcohesion obstacle avoidance and boundary conditions and strategic separationfor avoiding alien agents. The control strategy uses the local perception-basedbehavioral deviation vectors to guide each agents motion. Additionally thecontrol strategy incorporates a directionally-aware obstacle avoidancemechanism that prioritizes obstacles in the agents forward path. Simulationresults validate the effectiveness of this approach in creating flexibleadaptable and scalable flocking behavior. |


| Item |Content|
| --- |---|
|idx| 2409.09979v1 |
|title| Optimality Gap of Decentralized Submodular Maximization under Probabilistic Communication |
|authors| Joan VendrellSolmaz Kia
|links| http://arxiv.org/abs/2409.09979v1 |
|updated| 2024-09-16 04:18:16 UTC |
|summary| This paper considers the problem of decentralized submodular maximizationsubject to partition matroid constraint using a sequential greedy algorithmwith probabilistic inter-agent message-passing. We propose acommunication-aware framework where the probability of successful communicationbetween connected devices is considered. Our analysis introduces the notion ofthe probabilistic optimality gap highlighting its potential influence ondetermining the message-passing sequence based on the agents broadcastreliability and strategic decisions regarding agents that can broadcast theirmessages multiple times in a resource-limited environment. This work not onlycontributes theoretical insights but also has practical implications fordesigning and analyzing decentralized systems in uncertain communicationenvironments. A numerical example demonstrates the impact of our results. |


