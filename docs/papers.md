# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.07989v1 |
|title| Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding |
|authors| Yiwen TangJiaming LiuDong WangZhigang WangShanghang ZhangBin ZhaoXuelong Li
|links| http://arxiv.org/abs/2404.07989v1 |
|updated| 2024-04-11 17:59:45 UTC |
|summary| Large foundation models have recently emerged as a prominent focus ofinterest attaining superior performance in widespread scenarios. Due to thescarcity of 3D data many efforts have been made to adapt pre-trainedtransformers from vision to 3D domains. However such 2D-to-3D approaches arestill limited due to the potential loss of spatial geometries and highcomputation cost. More importantly their frameworks are mainly designed for 2Dmodels lacking a general any-to-3D paradigm. In this paper we introduceAny2Point a parameter-efficient method to empower any-modality large modelsvision language audio for 3D understanding. Given a frozen transformer fromany source modality we propose a 3D-to-any 1D or 2D virtual projectionstrategy that correlates the input 3D points to the original 1D or 2D positionswithin the source modality. This mechanism enables us to assign each 3D tokenwith a positional encoding paired with the pre-trained model which avoids 3Dgeometry loss caused by the true projection and better motivates thetransformer for 3D learning with 1D/2D positional priors. Then within eachtransformer block we insert an any-to-3D guided adapter module forparameter-efficient fine-tuning. The adapter incorporates prior spatialknowledge from the source modality to guide the local feature aggregation of 3Dtokens compelling the semantic adaption of any-modality transformers. Weconduct extensive experiments to showcase the effectiveness and efficiency ofour method. Code and models are released athttps://github.com/Ivan-Tang-3D/Any2Point. |


| Item |Content|
| --- |---|
|idx| 2404.07982v1 |
|title| Language Imbalance Can Boost Cross-lingual Generalisation |
|authors| Anton SchäferShauli RavfogelThomas HofmannTiago PimentelImanol Schlag
|links| http://arxiv.org/abs/2404.07982v1 |
|updated| 2024-04-11 17:58:05 UTC |
|summary| Multilinguality is crucial for extending recent advancements in languagemodelling to diverse linguistic communities. To maintain high performance whilerepresenting multiple languages multilingual models ideally alignrepresentations allowing what is learned in one language to generalise toothers. Prior research has emphasised the importance of parallel data andshared vocabulary elements as key factors for such alignment. In this study weinvestigate an unintuitive novel driver of cross-lingual generalisation:language imbalance. In controlled experiments on perfectly equivalent clonedlanguages we observe that the existence of a predominant language duringtraining boosts the performance of less frequent languages and leads tostronger alignment of model representations across languages. Furthermore wefind that this trend is amplified with scale: with large enough models or longenough training we observe that bilingual training data with a 90/10 languagesplit yields better performance on both languages than a balanced 50/50 split.Building on these insights we design training schemes that can improveperformance in all cloned languages even without altering the training data.As we extend our analysis to real languages we find that infrequent languagesstill benefit from frequent ones yet whether language imbalance causescross-lingual generalisation there is not conclusive. |


| Item |Content|
| --- |---|
|idx| 2404.07981v1 |
|title| Manipulating Large Language Models to Increase Product Visibility |
|authors| Aounon KumarHimabindu Lakkaraju
|links| http://arxiv.org/abs/2404.07981v1 |
|updated| 2024-04-11 17:57:32 UTC |
|summary| Large language models LLMs are increasingly being integrated into searchengines to provide natural language responses tailored to user queries.Customers and end-users are also becoming more dependent on these models forquick and easy purchase decisions. In this work we investigate whetherrecommendations from LLMs can be manipulated to enhance a products visibility.We demonstrate that adding a strategic text sequence STS -- a carefullycrafted message -- to a products information page can significantly increaseits likelihood of being listed as the LLMs top recommendation. To understandthe impact of STS we use a catalog of fictitious coffee machines and analyzeits effect on two target products: one that seldom appears in the LLMsrecommendations and another that usually ranks second. We observe that thestrategic text sequence significantly enhances the visibility of both productsby increasing their chances of appearing as the top recommendation. Thisability to manipulate LLM-generated search responses provides vendors with aconsiderable competitive advantage and has the potential to disrupt fair marketcompetition. Just as search engine optimization SEO revolutionized howwebpages are customized to rank higher in search engine results influencingLLM recommendations could profoundly impact content optimization for AI-drivensearch services. Code for our experiments is available athttps://github.com/aounon/llm-rank-optimizer. |


| Item |Content|
| --- |---|
|idx| 2404.07979v1 |
|title| LLoCO: Learning Long Contexts Offline |
|authors| Sijun TanXiuyu LiShishir PatilZiyang WuTianjun ZhangKurt KeutzerJoseph E. GonzalezRaluca Ada Popa
|links| http://arxiv.org/abs/2404.07979v1 |
|updated| 2024-04-11 17:57:22 UTC |
|summary| Processing long contexts remains a challenge for large language models LLMsdue to the quadratic computational and memory overhead of the self-attentionmechanism and the substantial KV cache sizes during generation. We propose anovel approach to address this problem by learning contexts offline throughcontext compression and in-domain parameter-efficient finetuning. Our methodenables an LLM to create a concise representation of the original context andefficiently retrieve relevant information to answer questions accurately. Weintroduce LLoCO a technique that combines context compression retrieval andparameter-efficient finetuning using LoRA. Our approach extends the effectivecontext window of a 4k token LLaMA2-7B model to handle up to 128k tokens. Weevaluate our approach on several long-context question-answering datasetsdemonstrating that LLoCO significantly outperforms in-context learning whileusing 30times fewer tokens during inference. LLoCO achieves up to7.62times speed-up and substantially reduces the cost of long documentquestion answering making it a promising solution for efficient long contextprocessing. Our code is publicly available athttps://github.com/jeffreysijuntan/lloco. |


| Item |Content|
| --- |---|
|idx| 2404.07972v1 |
|title| OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments |
|authors| Tianbao XieDanyang ZhangJixuan ChenXiaochuan LiSiheng ZhaoRuisheng CaoToh Jing HuaZhoujun ChengDongchan ShinFangyu LeiYitao LiuYiheng XuShuyan ZhouSilvio SavareseCaiming XiongVictor ZhongTao Yu
|links| http://arxiv.org/abs/2404.07972v1 |
|updated| 2024-04-11 17:56:05 UTC |
|summary| Autonomous agents that accomplish complex computer tasks with minimal humaninterventions have the potential to transform human-computer interactionsignificantly enhancing accessibility and productivity. However existingbenchmarks either lack an interactive environment or are limited toenvironments specific to certain applications or domains failing to reflectthe diverse and complex nature of real-world computer use thereby limiting thescope of tasks and agent scalability. To address this issue we introduceOSWorld the first-of-its-kind scalable real computer environment formultimodal agents supporting task setup execution-based evaluation andinteractive learning across various operating systems such as Ubuntu Windowsand macOS. OSWorld can serve as a unified integrated computer environment forassessing open-ended computer tasks that involve arbitrary applications.Building upon OSWorld we create a benchmark of 369 computer tasks involvingreal web and desktop apps in open domains OS file I/O and workflows spanningmultiple applications. Each task example is derived from real-world computeruse cases and includes a detailed initial state setup configuration and acustom execution-based evaluation script for reliable reproducible evaluation.Extensive evaluation of state-of-the-art LLM/VLM-based agents on OSWorldreveals significant deficiencies in their ability to serve as computerassistants. While humans can accomplish over 72.36 of the tasks the bestmodel achieves only 12.24 success primarily struggling with GUI grounding andoperational knowledge. Comprehensive analysis using OSWorld provides valuableinsights for developing multimodal generalist agents that were not possiblewith previous benchmarks. Our code environment baseline models and data arepublicly available at https://os-world.github.io. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.07990v1 |
|title| OpenBias: Open-set Bias Detection in Text-to-Image Generative Models |
|authors| Moreno D'IncàElia PeruzzoMassimiliano ManciniDejia XuVidit GoelXingqian XuZhangyang WangHumphrey ShiNicu Sebe
|links| http://arxiv.org/abs/2404.07990v1 |
|updated| 2024-04-11 17:59:56 UTC |
|summary| Text-to-image generative models are becoming increasingly popular andaccessible to the general public. As these models see large-scale deploymentsit is necessary to deeply investigate their safety and fairness to notdisseminate and perpetuate any kind of biases. However existing works focus ondetecting closed sets of biases defined a priori limiting the studies towell-known concepts. In this paper we tackle the challenge of open-set biasdetection in text-to-image generative models presenting OpenBias a newpipeline that identifies and quantifies the severity of biases agnosticallywithout access to any precompiled set. OpenBias has three stages. In the firstphase we leverage a Large Language Model LLM to propose biases given a setof captions. Secondly the target generative model produces images using thesame set of captions. Lastly a Vision Question Answering model recognizes thepresence and extent of the previously proposed biases. We study the behavior ofStable Diffusion 1.5 2 and XL emphasizing new biases never investigatedbefore. Via quantitative experiments we demonstrate that OpenBias agrees withcurrent closed-set bias detection methods and human judgement. |


| Item |Content|
| --- |---|
|idx| 2404.07989v1 |
|title| Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding |
|authors| Yiwen TangJiaming LiuDong WangZhigang WangShanghang ZhangBin ZhaoXuelong Li
|links| http://arxiv.org/abs/2404.07989v1 |
|updated| 2024-04-11 17:59:45 UTC |
|summary| Large foundation models have recently emerged as a prominent focus ofinterest attaining superior performance in widespread scenarios. Due to thescarcity of 3D data many efforts have been made to adapt pre-trainedtransformers from vision to 3D domains. However such 2D-to-3D approaches arestill limited due to the potential loss of spatial geometries and highcomputation cost. More importantly their frameworks are mainly designed for 2Dmodels lacking a general any-to-3D paradigm. In this paper we introduceAny2Point a parameter-efficient method to empower any-modality large modelsvision language audio for 3D understanding. Given a frozen transformer fromany source modality we propose a 3D-to-any 1D or 2D virtual projectionstrategy that correlates the input 3D points to the original 1D or 2D positionswithin the source modality. This mechanism enables us to assign each 3D tokenwith a positional encoding paired with the pre-trained model which avoids 3Dgeometry loss caused by the true projection and better motivates thetransformer for 3D learning with 1D/2D positional priors. Then within eachtransformer block we insert an any-to-3D guided adapter module forparameter-efficient fine-tuning. The adapter incorporates prior spatialknowledge from the source modality to guide the local feature aggregation of 3Dtokens compelling the semantic adaption of any-modality transformers. Weconduct extensive experiments to showcase the effectiveness and efficiency ofour method. Code and models are released athttps://github.com/Ivan-Tang-3D/Any2Point. |


| Item |Content|
| --- |---|
|idx| 2404.07987v1 |
|title| ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback |
|authors| Ming LiTaojiannan YangHuafeng KuangJie WuZhaoning WangXuefeng XiaoChen Chen
|links| http://arxiv.org/abs/2404.07987v1 |
|updated| 2024-04-11 17:59:09 UTC |
|summary| To enhance the controllability of text-to-image diffusion models existingefforts like ControlNet incorporated image-based conditional controls. In thispaper we reveal that existing methods still face significant challenges ingenerating images that align with the image conditional controls. To this endwe propose ControlNet a novel approach that improves controllable generationby explicitly optimizing pixel-level cycle consistency between generated imagesand conditional controls. Specifically for an input conditional control weuse a pre-trained discriminative reward model to extract the correspondingcondition of the generated images and then optimize the consistency lossbetween the input conditional control and extracted condition. Astraightforward implementation would be generating images from random noisesand then calculating the consistency loss but such an approach requiresstoring gradients for multiple sampling timesteps leading to considerable timeand memory costs. To address this we introduce an efficient reward strategythat deliberately disturbs the input images by adding noise and then uses thesingle-step denoised images for reward fine-tuning. This avoids the extensivecosts associated with image sampling allowing for more efficient rewardfine-tuning. Extensive experiments show that ControlNet significantlyimproves controllability under various conditional controls. For example itachieves improvements over ControlNet by 7.9 mIoU 13.4 SSIM and 7.6 RMSErespectively for segmentation mask line-art edge and depth conditions. |


| Item |Content|
| --- |---|
|idx| 2404.07981v1 |
|title| Manipulating Large Language Models to Increase Product Visibility |
|authors| Aounon KumarHimabindu Lakkaraju
|links| http://arxiv.org/abs/2404.07981v1 |
|updated| 2024-04-11 17:57:32 UTC |
|summary| Large language models LLMs are increasingly being integrated into searchengines to provide natural language responses tailored to user queries.Customers and end-users are also becoming more dependent on these models forquick and easy purchase decisions. In this work we investigate whetherrecommendations from LLMs can be manipulated to enhance a products visibility.We demonstrate that adding a strategic text sequence STS -- a carefullycrafted message -- to a products information page can significantly increaseits likelihood of being listed as the LLMs top recommendation. To understandthe impact of STS we use a catalog of fictitious coffee machines and analyzeits effect on two target products: one that seldom appears in the LLMsrecommendations and another that usually ranks second. We observe that thestrategic text sequence significantly enhances the visibility of both productsby increasing their chances of appearing as the top recommendation. Thisability to manipulate LLM-generated search responses provides vendors with aconsiderable competitive advantage and has the potential to disrupt fair marketcompetition. Just as search engine optimization SEO revolutionized howwebpages are customized to rank higher in search engine results influencingLLM recommendations could profoundly impact content optimization for AI-drivensearch services. Code for our experiments is available athttps://github.com/aounon/llm-rank-optimizer. |


| Item |Content|
| --- |---|
|idx| 2404.07979v1 |
|title| LLoCO: Learning Long Contexts Offline |
|authors| Sijun TanXiuyu LiShishir PatilZiyang WuTianjun ZhangKurt KeutzerJoseph E. GonzalezRaluca Ada Popa
|links| http://arxiv.org/abs/2404.07979v1 |
|updated| 2024-04-11 17:57:22 UTC |
|summary| Processing long contexts remains a challenge for large language models LLMsdue to the quadratic computational and memory overhead of the self-attentionmechanism and the substantial KV cache sizes during generation. We propose anovel approach to address this problem by learning contexts offline throughcontext compression and in-domain parameter-efficient finetuning. Our methodenables an LLM to create a concise representation of the original context andefficiently retrieve relevant information to answer questions accurately. Weintroduce LLoCO a technique that combines context compression retrieval andparameter-efficient finetuning using LoRA. Our approach extends the effectivecontext window of a 4k token LLaMA2-7B model to handle up to 128k tokens. Weevaluate our approach on several long-context question-answering datasetsdemonstrating that LLoCO significantly outperforms in-context learning whileusing 30times fewer tokens during inference. LLoCO achieves up to7.62times speed-up and substantially reduces the cost of long documentquestion answering making it a promising solution for efficient long contextprocessing. Our code is publicly available athttps://github.com/jeffreysijuntan/lloco. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.07989v1 |
|title| Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding |
|authors| Yiwen TangJiaming LiuDong WangZhigang WangShanghang ZhangBin ZhaoXuelong Li
|links| http://arxiv.org/abs/2404.07989v1 |
|updated| 2024-04-11 17:59:45 UTC |
|summary| Large foundation models have recently emerged as a prominent focus ofinterest attaining superior performance in widespread scenarios. Due to thescarcity of 3D data many efforts have been made to adapt pre-trainedtransformers from vision to 3D domains. However such 2D-to-3D approaches arestill limited due to the potential loss of spatial geometries and highcomputation cost. More importantly their frameworks are mainly designed for 2Dmodels lacking a general any-to-3D paradigm. In this paper we introduceAny2Point a parameter-efficient method to empower any-modality large modelsvision language audio for 3D understanding. Given a frozen transformer fromany source modality we propose a 3D-to-any 1D or 2D virtual projectionstrategy that correlates the input 3D points to the original 1D or 2D positionswithin the source modality. This mechanism enables us to assign each 3D tokenwith a positional encoding paired with the pre-trained model which avoids 3Dgeometry loss caused by the true projection and better motivates thetransformer for 3D learning with 1D/2D positional priors. Then within eachtransformer block we insert an any-to-3D guided adapter module forparameter-efficient fine-tuning. The adapter incorporates prior spatialknowledge from the source modality to guide the local feature aggregation of 3Dtokens compelling the semantic adaption of any-modality transformers. Weconduct extensive experiments to showcase the effectiveness and efficiency ofour method. Code and models are released athttps://github.com/Ivan-Tang-3D/Any2Point. |


| Item |Content|
| --- |---|
|idx| 2404.07987v1 |
|title| ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback |
|authors| Ming LiTaojiannan YangHuafeng KuangJie WuZhaoning WangXuefeng XiaoChen Chen
|links| http://arxiv.org/abs/2404.07987v1 |
|updated| 2024-04-11 17:59:09 UTC |
|summary| To enhance the controllability of text-to-image diffusion models existingefforts like ControlNet incorporated image-based conditional controls. In thispaper we reveal that existing methods still face significant challenges ingenerating images that align with the image conditional controls. To this endwe propose ControlNet a novel approach that improves controllable generationby explicitly optimizing pixel-level cycle consistency between generated imagesand conditional controls. Specifically for an input conditional control weuse a pre-trained discriminative reward model to extract the correspondingcondition of the generated images and then optimize the consistency lossbetween the input conditional control and extracted condition. Astraightforward implementation would be generating images from random noisesand then calculating the consistency loss but such an approach requiresstoring gradients for multiple sampling timesteps leading to considerable timeand memory costs. To address this we introduce an efficient reward strategythat deliberately disturbs the input images by adding noise and then uses thesingle-step denoised images for reward fine-tuning. This avoids the extensivecosts associated with image sampling allowing for more efficient rewardfine-tuning. Extensive experiments show that ControlNet significantlyimproves controllability under various conditional controls. For example itachieves improvements over ControlNet by 7.9 mIoU 13.4 SSIM and 7.6 RMSErespectively for segmentation mask line-art edge and depth conditions. |


| Item |Content|
| --- |---|
|idx| 2404.07983v1 |
|title| Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Representation Learning |
|authors| Simon SchrodiDavid T. HoffmannMax ArgusVolker FischerThomas Brox
|links| http://arxiv.org/abs/2404.07983v1 |
|updated| 2024-04-11 17:58:06 UTC |
|summary| Contrastive vision-language models like CLIP have gained popularity for theirversatile applicable learned representations in various downstream tasks.Despite their successes in some tasks like zero-shot image recognition theyalso perform surprisingly poor on other tasks like attribute detection.Previous work has attributed these challenges to the modality gap a separationof image and text in the shared representation space and a bias towardsobjects over other factors such as attributes. In this work we investigateboth phenomena. We find that only a few embedding dimensions drive the modalitygap. Further we propose a measure for object bias and find that object biasdoes not lead to worse performance on other concepts such as attributes. Butwhat leads to the emergence of the modality gap and object bias To answer thisquestion we carefully designed an experimental setting which allows us tocontrol the amount of shared information between the modalities. This revealedthat the driving factor behind both the modality gap and the object bias isthe information imbalance between images and captions. |


| Item |Content|
| --- |---|
|idx| 2404.07982v1 |
|title| Language Imbalance Can Boost Cross-lingual Generalisation |
|authors| Anton SchäferShauli RavfogelThomas HofmannTiago PimentelImanol Schlag
|links| http://arxiv.org/abs/2404.07982v1 |
|updated| 2024-04-11 17:58:05 UTC |
|summary| Multilinguality is crucial for extending recent advancements in languagemodelling to diverse linguistic communities. To maintain high performance whilerepresenting multiple languages multilingual models ideally alignrepresentations allowing what is learned in one language to generalise toothers. Prior research has emphasised the importance of parallel data andshared vocabulary elements as key factors for such alignment. In this study weinvestigate an unintuitive novel driver of cross-lingual generalisation:language imbalance. In controlled experiments on perfectly equivalent clonedlanguages we observe that the existence of a predominant language duringtraining boosts the performance of less frequent languages and leads tostronger alignment of model representations across languages. Furthermore wefind that this trend is amplified with scale: with large enough models or longenough training we observe that bilingual training data with a 90/10 languagesplit yields better performance on both languages than a balanced 50/50 split.Building on these insights we design training schemes that can improveperformance in all cloned languages even without altering the training data.As we extend our analysis to real languages we find that infrequent languagesstill benefit from frequent ones yet whether language imbalance causescross-lingual generalisation there is not conclusive. |


| Item |Content|
| --- |---|
|idx| 2404.07979v1 |
|title| LLoCO: Learning Long Contexts Offline |
|authors| Sijun TanXiuyu LiShishir PatilZiyang WuTianjun ZhangKurt KeutzerJoseph E. GonzalezRaluca Ada Popa
|links| http://arxiv.org/abs/2404.07979v1 |
|updated| 2024-04-11 17:57:22 UTC |
|summary| Processing long contexts remains a challenge for large language models LLMsdue to the quadratic computational and memory overhead of the self-attentionmechanism and the substantial KV cache sizes during generation. We propose anovel approach to address this problem by learning contexts offline throughcontext compression and in-domain parameter-efficient finetuning. Our methodenables an LLM to create a concise representation of the original context andefficiently retrieve relevant information to answer questions accurately. Weintroduce LLoCO a technique that combines context compression retrieval andparameter-efficient finetuning using LoRA. Our approach extends the effectivecontext window of a 4k token LLaMA2-7B model to handle up to 128k tokens. Weevaluate our approach on several long-context question-answering datasetsdemonstrating that LLoCO significantly outperforms in-context learning whileusing 30times fewer tokens during inference. LLoCO achieves up to7.62times speed-up and substantially reduces the cost of long documentquestion answering making it a promising solution for efficient long contextprocessing. Our code is publicly available athttps://github.com/jeffreysijuntan/lloco. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.07992v1 |
|title| GoMVS: Geometrically Consistent Cost Aggregation for Multi-View Stereo |
|authors| Jiang WuRui LiHaofei XuWenxun ZhaoYu ZhuJinqiu SunYanning Zhang
|links| http://arxiv.org/abs/2404.07992v1 |
|updated| 2024-04-11 17:59:59 UTC |
|summary| Matching cost aggregation plays a fundamental role in learning-basedmulti-view stereo networks. However directly aggregating adjacent costs canlead to suboptimal results due to local geometric inconsistency. Relatedmethods either seek selective aggregation or improve aggregated depth in the 2Dspace both are unable to handle geometric inconsistency in the cost volumeeffectively. In this paper we propose GoMVS to aggregate geometricallyconsistent costs yielding better utilization of adjacent geometries. Morespecifically we correspond and propagate adjacent costs to the reference pixelby leveraging the local geometric smoothness in conjunction with surfacenormals. We achieve this by the geometric consistent propagation GCP module.It computes the correspondence from the adjacent depth hypothesis space to thereference depth space using surface normals then uses the correspondence topropagate adjacent costs to the reference geometry followed by a convolutionfor aggregation. Our method achieves new state-of-the-art performance on DTUTanks  Temple and ETH3D datasets. Notably our method ranks 1st on the Tanks Temple Advanced benchmark. |


| Item |Content|
| --- |---|
|idx| 2404.07993v1 |
|title| Connecting NeRFs, Images, and Text |
|authors| Francesco BalleriniPierluigi Zama RamirezRoberto MirabellaSamuele SaltiLuigi Di Stefano
|links| http://arxiv.org/abs/2404.07993v1 |
|updated| 2024-04-11 17:59:59 UTC |
|summary| Neural Radiance Fields NeRFs have emerged as a standard framework forrepresenting 3D scenes and objects introducing a novel data type forinformation exchange and storage. Concurrently significant progress has beenmade in multimodal representation learning for text and image data. This paperexplores a novel research direction that aims to connect the NeRF modality withother modalities similar to established methodologies for images and text. Tothis end we propose a simple framework that exploits pre-trained models forNeRF representations alongside multimodal models for text and image processing.Our framework learns a bidirectional mapping between NeRF embeddings and thoseobtained from corresponding images and text. This mapping unlocks several noveland useful applications including NeRF zero-shot classification and NeRFretrieval from images or text. |


| Item |Content|
| --- |---|
|idx| 2404.07991v1 |
|title| GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh |
|authors| Jing WenXiaoming ZhaoZhongzheng RenAlexander G. SchwingShenlong Wang
|links| http://arxiv.org/abs/2404.07991v1 |
|updated| 2024-04-11 17:59:57 UTC |
|summary| We introduce GoMAvatar a novel approach for real-time memory-efficienthigh-quality animatable human modeling. GoMAvatar takes as input a singlemonocular video to create a digital avatar capable of re-articulation in newposes and real-time rendering from novel viewpoints while seamlesslyintegrating with rasterization-based graphics pipelines. Central to our methodis the Gaussians-on-Mesh representation a hybrid 3D model combining renderingquality and speed of Gaussian splatting with geometry modeling andcompatibility of deformable meshes. We assess GoMAvatar on ZJU-MoCap data andvarious YouTube videos. GoMAvatar matches or surpasses current monocular humanmodeling algorithms in rendering quality and significantly outperforms them incomputational efficiency 43 FPS while being memory-efficient 3.63 MB persubject. |


| Item |Content|
| --- |---|
|idx| 2404.07990v1 |
|title| OpenBias: Open-set Bias Detection in Text-to-Image Generative Models |
|authors| Moreno D'IncàElia PeruzzoMassimiliano ManciniDejia XuVidit GoelXingqian XuZhangyang WangHumphrey ShiNicu Sebe
|links| http://arxiv.org/abs/2404.07990v1 |
|updated| 2024-04-11 17:59:56 UTC |
|summary| Text-to-image generative models are becoming increasingly popular andaccessible to the general public. As these models see large-scale deploymentsit is necessary to deeply investigate their safety and fairness to notdisseminate and perpetuate any kind of biases. However existing works focus ondetecting closed sets of biases defined a priori limiting the studies towell-known concepts. In this paper we tackle the challenge of open-set biasdetection in text-to-image generative models presenting OpenBias a newpipeline that identifies and quantifies the severity of biases agnosticallywithout access to any precompiled set. OpenBias has three stages. In the firstphase we leverage a Large Language Model LLM to propose biases given a setof captions. Secondly the target generative model produces images using thesame set of captions. Lastly a Vision Question Answering model recognizes thepresence and extent of the previously proposed biases. We study the behavior ofStable Diffusion 1.5 2 and XL emphasizing new biases never investigatedbefore. Via quantitative experiments we demonstrate that OpenBias agrees withcurrent closed-set bias detection methods and human judgement. |


| Item |Content|
| --- |---|
|idx| 2404.07989v1 |
|title| Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding |
|authors| Yiwen TangJiaming LiuDong WangZhigang WangShanghang ZhangBin ZhaoXuelong Li
|links| http://arxiv.org/abs/2404.07989v1 |
|updated| 2024-04-11 17:59:45 UTC |
|summary| Large foundation models have recently emerged as a prominent focus ofinterest attaining superior performance in widespread scenarios. Due to thescarcity of 3D data many efforts have been made to adapt pre-trainedtransformers from vision to 3D domains. However such 2D-to-3D approaches arestill limited due to the potential loss of spatial geometries and highcomputation cost. More importantly their frameworks are mainly designed for 2Dmodels lacking a general any-to-3D paradigm. In this paper we introduceAny2Point a parameter-efficient method to empower any-modality large modelsvision language audio for 3D understanding. Given a frozen transformer fromany source modality we propose a 3D-to-any 1D or 2D virtual projectionstrategy that correlates the input 3D points to the original 1D or 2D positionswithin the source modality. This mechanism enables us to assign each 3D tokenwith a positional encoding paired with the pre-trained model which avoids 3Dgeometry loss caused by the true projection and better motivates thetransformer for 3D learning with 1D/2D positional priors. Then within eachtransformer block we insert an any-to-3D guided adapter module forparameter-efficient fine-tuning. The adapter incorporates prior spatialknowledge from the source modality to guide the local feature aggregation of 3Dtokens compelling the semantic adaption of any-modality transformers. Weconduct extensive experiments to showcase the effectiveness and efficiency ofour method. Code and models are released athttps://github.com/Ivan-Tang-3D/Any2Point. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.07937v1 |
|title| Rate-Optimal Non-Asymptotics for the Quadratic Prediction Error Method |
|authors| Charis StamouliIngvar ZiemannGeorge J. Pappas
|links| http://arxiv.org/abs/2404.07937v1 |
|updated| 2024-04-11 17:36:28 UTC |
|summary| We study the quadratic prediction error method -- i.e. nonlinear leastsquares -- for a class of time-varying parametric predictor models satisfying acertain identifiability condition. While this method is known to asymptoticallyachieve the optimal rate for a wide range of problems there have been nonon-asymptotic results matching these optimal rates outside of a select fewtypically linear model classes. By leveraging modern tools from learning withdependent data we provide the first rate-optimal non-asymptotic analysis ofthis method for our more general setting of nonlinearly parametrized modelclasses. Moreover we show that our results can be applied to a particularclass of identifiable AutoRegressive Moving Average ARMA models resulting inthe first optimal non-asymptotic rates for identification of ARMA models. |


| Item |Content|
| --- |---|
|idx| 2404.07864v1 |
|title| Inferring Change Points in High-Dimensional Linear Regression via Approximate Message Passing |
|authors| Gabriel ArpinoXiaoqi LiuRamji Venkataramanan
|links| http://arxiv.org/abs/2404.07864v1 |
|updated| 2024-04-11 15:57:12 UTC |
|summary| We consider the problem of localizing change points in high-dimensionallinear regression. We propose an Approximate Message Passing AMP algorithmfor estimating both the signals and the change point locations. AssumingGaussian covariates we give an exact asymptotic characterization of itsestimation performance in the limit where the number of samples growsproportionally to the signal dimension. Our algorithm can be tailored toexploit any prior information on the signal noise and change points. It alsoenables uncertainty quantification in the form of an efficiently computableapproximate posterior distribution whose asymptotic form we characterizeexactly. We validate our theory via numerical experiments and demonstrate thefavorable performance of our estimators on both synthetic data and images. |


| Item |Content|
| --- |---|
|idx| 2404.07849v1 |
|title| Overparameterized Multiple Linear Regression as Hyper-Curve Fitting |
|authors| E. AtzaN. Budko
|links| http://arxiv.org/abs/2404.07849v1 |
|updated| 2024-04-11 15:43:11 UTC |
|summary| The paper shows that the application of the fixed-effect multiple linearregression model to an overparameterized dataset is equivalent to fitting thedata with a hyper-curve parameterized by a single scalar parameter. Thisequivalence allows for a predictor-focused approach where each predictor isdescribed by a function of the chosen parameter. It is proven that a linearmodel will produce exact predictions even in the presence of nonlineardependencies that violate the model assumptions. Parameterization in terms ofthe dependent variable and the monomial basis in the predictor function spaceare applied here to both synthetic and experimental data. The hyper-curveapproach is especially suited for the regularization of problems with noise inpredictor variables and can be used to remove noisy and improper predictorsfrom the model. |


| Item |Content|
| --- |---|
|idx| 2404.07815v1 |
|title| Post-Hoc Reversal: Are We Selecting Models Prematurely? |
|authors| Rishabh RanjanSaurabh GargMrigank RamanCarlos GuestrinZachary Chase Lipton
|links| http://arxiv.org/abs/2404.07815v1 |
|updated| 2024-04-11 14:58:19 UTC |
|summary| Trained models are often composed with post-hoc transforms such astemperature scaling TS ensembling and stochastic weight averaging SWA toimprove performance robustness uncertainty estimation etc. However suchtransforms are typically applied only after the base models have already beenfinalized by standard means. In this paper we challenge this practice with anextensive empirical study. In particular we demonstrate a phenomenon that wecall post-hoc reversal where performance trends are reversed after applyingthese post-hoc transforms. This phenomenon is especially prominent inhigh-noise settings. For example while base models overfit badly early intraining both conventional ensembling and SWA favor base models trained formore epochs. Post-hoc reversal can also suppress the appearance of doubledescent and mitigate mismatches between test loss and test error seen in basemodels. Based on our findings we propose post-hoc selection a simpletechnique whereby post-hoc metrics inform model development decisions such asearly stopping checkpointing and broader hyperparameter choices. Ourexperimental analyses span real-world vision language tabular and graphdatasets from domains like satellite imaging language modeling censusprediction and social network analysis. On an LLM instruction tuning datasetpost-hoc selection results in  1.5x MMLU improvement compared to naiveselection. Code is available athttps://github.com/rishabh-ranjan/post-hoc-reversal. |


| Item |Content|
| --- |---|
|idx| 2404.07778v1 |
|title| Quality check of a sample partition using multinomial distribution |
|authors| Soumita Modak
|links| http://arxiv.org/abs/2404.07778v1 |
|updated| 2024-04-11 14:14:58 UTC |
|summary| In this paper we advocate a novel measure for the purpose of checking thequality of a cluster partition for a sample into several distinct classes andthus determine the unknown value for the true number of clusters prevailingthe provided set of data. Our objective leads us to the development of anapproach through applying the multinomial distribution to the distances of datamembers clustered in a group from their respective cluster representatives.This procedure is carried out independently for each of the clusters and theconcerned statistics are combined together to design our targeted measure.Individual clusters separately possess the category-wise probabilities whichcorrespond to different positions of its members in the cluster with respect toa typical member in the form of cluster-centroid medoid or mode referred toas the corresponding cluster representative. Our method is robust in the sensethat it is distribution-free since this is devised irrespective of the parentdistribution of the underlying sample. It fulfills one of the rare covetedqualities present in the existing cluster accuracy measures of having thecapability to investigate whether the assigned sample owns any inherentclusters other than a single group of all members or not. Our measures simpleconcept easy algorithm fast runtime good performance and wide usefulnessdemonstrated through extensive simulation and diverse case-studies make itappealing. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.07934v1 |
|title| Goal Recognition via Linear Programming |
|authors| Felipe MeneguzziLuísa R. de A. SantosRamon Fraga PereiraAndré G. Pereira
|links| http://arxiv.org/abs/2404.07934v1 |
|updated| 2024-04-11 17:34:35 UTC |
|summary| Goal Recognition is the task by which an observer aims to discern the goalsthat correspond to plans that comply with the perceived behavior of subjectagents given as a sequence of observations. Research on Goal Recognition asPlanning encompasses reasoning about the model of a planning task theobservations and the goals using planning techniques resulting in veryefficient recognition approaches. In this article we design novel recognitionapproaches that rely on the Operator-Counting framework proposing newconstraints and analyze their constraints properties both theoretically andempirically. The Operator-Counting framework is a technique that efficientlycomputes heuristic estimates of cost-to-goal using Integer/Linear ProgrammingIP/LP. In the realm of theory we prove that the new constraints providelower bounds on the cost of plans that comply with observations. We alsoprovide an extensive empirical evaluation to assess how the new constraintsimprove the quality of the solution and we found that they are especiallyinformed in deciding which goals are unlikely to be part of the solution. Ournovel recognition approaches have two pivotal advantages: first they employnew IP/LP constraints for efficiently recognizing goals second we show howthe new IP/LP constraints can improve the recognition of goals under bothpartial and noisy observability. |


| Item |Content|
| --- |---|
|idx| 2404.07926v1 |
|title| Leveraging Large Language Models (LLMs) to Support Collaborative Human-AI Online Risk Data Annotation |
|authors| Jinkyung ParkPamela WisniewskiVivek Singh
|links| http://arxiv.org/abs/2404.07926v1 |
|updated| 2024-04-11 17:20:57 UTC |
|summary| In this position paper we discuss the potential for leveraging LLMs asinteractive research tools to facilitate collaboration between human coders andAI to effectively annotate online risk data at scale. Collaborative human-AIlabeling is a promising approach to annotating large-scale and complex data forvarious tasks. Yet tools and methods to support effective human-AIcollaboration for data annotation are under-studied. This gap is pertinentbecause co-labeling tasks need to support a two-way interactive discussion thatcan add nuance and context particularly in the context of online risk whichis highly subjective and contextualized. Therefore we provide some of theearly benefits and challenges of using LLMs-based tools for risk annotation andsuggest future directions for the HCI research community to leverage LLMs asresearch tools to facilitate human-AI collaboration in contextualized onlinedata annotation. Our research interests align very well with the purposes ofthe LLMs as Research Tools workshop to identify ongoing applications andchallenges of using LLMs to work with data in HCI research. We anticipatelearning valuable insights from organizers and participants into how LLMs canhelp reshape the HCI communitys methods for working with data. |


| Item |Content|
| --- |---|
|idx| 2404.07901v1 |
|title| Snake Story: Exploring Game Mechanics for Mixed-Initiative Co-creative Storytelling Games |
|authors| Daijin YangErica KleinmanGiovanni Maria TroianoElina TochilnikovaCasper Harteveld
|links| http://dx.doi.org/10.1145/3649921.3649996 |
|updated| 2024-04-11 16:40:24 UTC |
|summary| Mixed-initiative co-creative storytelling games have existed for some time asa way to merge storytelling with play. However modern mixed-initiativeco-creative storytelling games predominantly prioritize story creation overgameplay mechanics which might not resonate with all players. As such thereis untapped potential for creating mixed-initiative games with more complexmechanics in which players can engage with both co-creation and gameplay goals.To explore the potential of more prominent gameplay in mixed-initiativeco-creative storytelling games we created Snake Story a variation of theclassic Snake game featuring a human-AI co-writing element. To explore howplayers interact with the mixed-initiative game we conducted a qualitativeplaytest with 11 participants. Analysis of both think-aloud and interview datarevealed that players strategies and experiences were affected by theirperception of Snake Story as either a collaborative tool a traditional gameor a combination of both. Based on these findings we present designconsiderations for future development in mixed-initiative co-creative gaming. |


| Item |Content|
| --- |---|
|idx| 2404.07883v1 |
|title| Apprentice Tutor Builder: A Platform For Users to Create and Personalize Intelligent Tutors |
|authors| Glen SmithAdit GuptaChristopher MacLellan
|links| http://arxiv.org/abs/2404.07883v1 |
|updated| 2024-04-11 16:14:23 UTC |
|summary| Intelligent tutoring systems ITS are effective for improving studentslearning outcomes. However their development is often complex time-consumingand requires specialized programming and tutor design knowledge thus hinderingtheir widespread application and personalization. We present the ApprenticeTutor Builder ATB  a platform that simplifies tutor creation andpersonalization. Instructors can utilize ATBs drag-and-drop tool to buildtutor interfaces. Instructors can then interactively train the tutorsunderlying AI agent to produce expert models that can solve problems. Trainingis achieved via using multiple interaction modalities including demonstrationsfeedback and user labels. We conducted a user study with 14 instructors toevaluate the effectiveness of ATBs design with end users. We found that usersenjoyed the flexibility of the interface builder and ease and speed of agentteaching but often desired additional time-saving features. With theseinsights we identified a set of design recommendations for our platform andothers that utilize interactive AI agents for tutor creation and customization. |


| Item |Content|
| --- |---|
|idx| 2404.07865v1 |
|title| The Dance of Logic and Unpredictability: Examining the Predictability of User Behavior on Visual Analytics Tasks |
|authors| Alvitta Ottley
|links| http://dx.doi.org/10.5220/0012671100003660 |
|updated| 2024-04-11 15:57:18 UTC |
|summary| The quest to develop intelligent visual analytics VA systems capable ofcollaborating and naturally interacting with humans presents a multifaceted andintriguing challenge. VA systems designed for collaboration must adeptlynavigate a complex landscape filled with the subtleties and unpredictabilitiesthat characterize human behavior. However it is noteworthy that scenariosexist where human behavior manifests predictably. These scenarios typicallyinvolve routine actions or present a limited range of choices. This paperdelves into the predictability of user behavior in the context of visualanalytics tasks. It offers an evidence-based discussion on the circumstancesunder which predicting user behavior is feasible and those where it proveschallenging. We conclude with a forward-looking discussion of the future worknecessary to cultivate more synergistic and efficient partnerships betweenhumans and the VA system. This exploration is not just about understanding ourcurrent capabilities and limitations in mirroring human behavior but also aboutenvisioning and paving the way for a future where human-machine interaction ismore intuitive and productive. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.07902v1 |
|title| Q-ITAGS: Quality-Optimized Spatio-Temporal Heterogeneous Task Allocation with a Time Budget |
|authors| Glen NevilleJiazhen LiuSonia ChernovaHarish Ravichandar
|links| http://arxiv.org/abs/2404.07902v1 |
|updated| 2024-04-11 16:41:08 UTC |
|summary| Complex multi-objective missions require the coordination of heterogeneousrobots at multiple inter-connected levels such as coalition formationscheduling and motion planning. The associated challenges are exacerbated whensolutions to these interconnected problems need to both maximize taskperformance and respect practical constraints on time and resources. In thiswork we formulate a new class of spatio-temporal heterogeneous task allocationproblems that consider these complexities. We contribute a novel frameworknamed Quality-Optimized Incremental Task Allocation Graph Search Q-ITAGS tosolve such problems. Q-ITAGS builds upon our prior work in trait-basedcoordination and offers a flexible interleaved framework that i explicitlymodels and optimizes the effect of collective capabilities on task performancevia learnable trait-quality maps and ii respects both resource constraintsand spatio-temporal constraints including a user-specified time budget i.e.maximum makespan. In addition to algorithmic contributions we derivetheoretical suboptimality bounds in terms of task performance that varies as afunction of a single hyperparameter. Our detailed experiments involving asimulated emergency response task and a real-world video game dataset revealthat i Q-ITAGS results in superior team performance compared to astate-of-the-art method while also respecting complex spatio-temporal andresource constraints ii Q-ITAGS efficiently learns trait-quality maps toenable effective trade-off between task performance and resource constraintsand iii Q-ITAGS suboptimality bounds consistently hold in practice. |


| Item |Content|
| --- |---|
|idx| 2404.07838v1 |
|title| The Role of Confidence for Trust-based Resilient Consensus (Extended Version) |
|authors| Luca Ballotta. Michal Yemini
|links| http://arxiv.org/abs/2404.07838v1 |
|updated| 2024-04-11 15:27:14 UTC |
|summary| We consider a multi-agent system where agents aim to achieve a consensusdespite interactions with malicious agents that communicate misleadinginformation. Physical channels supporting communication in cyberphysicalsystems offer attractive opportunities to detect malicious agentsnevertheless trustworthiness indications coming from the channel are subjectto uncertainty and need to be treated with this in mind. We propose a resilientconsensus protocol that incorporates trust observations from the channel andweighs them with a parameter that accounts for how confident an agent isregarding its understanding of the legitimacy of other agents in the networkwith no need for the initial observation window T_0 that has been utilized inprevious works. Analytical and numerical results show that i our protocolachieves a resilient consensus in the presence of malicious agents and ii thesteady-state deviation from nominal consensus can be minimized by a suitablechoice of the confidence parameter that depends on the statistics of trustobservations. |


| Item |Content|
| --- |---|
|idx| 2404.07559v1 |
|title| Differentially Private Reinforcement Learning with Self-Play |
|authors| Dan QiaoYu-Xiang Wang
|links| http://arxiv.org/abs/2404.07559v1 |
|updated| 2024-04-11 08:42:51 UTC |
|summary| We study the problem of multi-agent reinforcement learning multi-agent RLwith differential privacy DP constraints. This is well-motivated by variousreal-world applications involving sensitive data where it is critical toprotect users private information. We first extend the definitions of Joint DPJDP and Local DP LDP to two-player zero-sum episodic Markov Games whereboth definitions ensure trajectory-wise privacy protection. Then we design aprovably efficient algorithm based on optimistic Nash value iteration andprivatization of Bernstein-type bonuses. The algorithm is able to satisfy JDPand LDP requirements when instantiated with appropriate privacy mechanisms.Furthermore for both notions of DP our regret bound generalizes the bestknown result under the single-agent RL case while our regret could also reduceto the best known result for multi-agent RL without privacy constraints. To thebest of our knowledge these are the first line of results towardsunderstanding trajectory-wise privacy protection in multi-agent RL. |


| Item |Content|
| --- |---|
|idx| 2404.07456v1 |
|title| WESE: Weak Exploration to Strong Exploitation for LLM Agents |
|authors| Xu HuangWeiwen LiuXiaolong ChenXingmei WangDefu LianYasheng WangRuiming TangEnhong Chen
|links| http://arxiv.org/abs/2404.07456v1 |
|updated| 2024-04-11 03:31:54 UTC |
|summary| Recently large language models LLMs have demonstrated remarkable potentialas an intelligent agent. However existing researches mainly focus on enhancingthe agents reasoning or decision-making abilities through well-designed promptengineering or task-specific fine-tuning ignoring the procedure of explorationand exploitation. When addressing complex tasks within open-world interactiveenvironments these methods exhibit limitations. Firstly the lack of globalinformation of environments leads to greedy decisions resulting in sub-optimalsolutions. On the other hand irrelevant information acquired from theenvironment not only adversely introduces noise but also incurs additionalcost. This paper proposes a novel approach Weak Exploration to StrongExploitation WESE to enhance LLM agents in solving open-world interactivetasks. Concretely WESE involves decoupling the exploration and exploitationprocess employing a cost-effective weak agent to perform exploration tasks forglobal knowledge. A knowledge graph-based strategy is then introduced to storethe acquired knowledge and extract task-relevant knowledge enhancing thestronger agent in success rate and efficiency for the exploitation task. Ourapproach is flexible enough to incorporate diverse tasks and obtainssignificant improvements in both success rates and efficiency across fourinteractive benchmarks. |


| Item |Content|
| --- |---|
|idx| 2404.06975v1 |
|title| Multi-Agent Soft Actor-Critic with Global Loss for Autonomous Mobility-on-Demand Fleet Control |
|authors| Zeno WoywoodJasper I. WiltfangJulius LuyTobias EndersMaximilian Schiffer
|links| http://arxiv.org/abs/2404.06975v1 |
|updated| 2024-04-10 13:49:20 UTC |
|summary| We study a sequential decision-making problem for a profit-maximizingoperator of an Autonomous Mobility-on-Demand system. Optimizing a centraloperators vehicle-to-request dispatching policy requires efficient andeffective fleet control strategies. To this end we employ a multi-agent SoftActor-Critic algorithm combined with weighted bipartite matching. We propose anovel vehicle-based algorithm architecture and adapt the critics loss functionto appropriately consider global actions. Furthermore we extend our algorithmto incorporate rebalancing capabilities. Through numerical experiments we showthat our approach outperforms state-of-the-art benchmarks by up to 12.9 fordispatching and up to 38.9 with integrated rebalancing. |


