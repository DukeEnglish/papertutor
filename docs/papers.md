# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.12390v1 |
|title| BLINK: Multimodal Large Language Models Can See but Not Perceive |
|authors| Xingyu FuYushi HuBangzheng LiYu FengHaoyu WangXudong LinDan RothNoah A. SmithWei-Chiu MaRanjay Krishna
|links| http://arxiv.org/abs/2404.12390v1 |
|updated| 2024-04-18 17:59:54 UTC |
|summary| We introduce Blink a new benchmark for multimodal language models LLMsthat focuses on core visual perception abilities not found in otherevaluations. Most of the Blink tasks can be solved by humans within a blinke.g. relative depth estimation visual correspondence forensics detectionand multi-view reasoning. However we find these perception-demanding taskscast significant challenges for current multimodal LLMs because they resistmediation through natural language. Blink reformats 14 classic computer visiontasks into 3807 multiple-choice questions paired with single or multipleimages and visual prompting. While humans get 95.70 accuracy on average Blinkis surprisingly challenging for existing multimodal LLMs: even thebest-performing GPT-4V and Gemini achieve accuracies of 51.26 and 45.72 only13.17 and 7.63 higher than random guessing indicating that such perceptionabilities have not emerged yet in recent multimodal LLMs. Our analysis alsohighlights that specialist CV models could solve these problems much bettersuggesting potential pathways for future improvements. We believe Blink willstimulate the community to help multimodal LLMs catch up with human-levelvisual perception. |


| Item |Content|
| --- |---|
|idx| 2404.12387v1 |
|title| Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models |
|authors| Aitor OrmazabalChe ZhengCyprien de Masson d'AutumeDani YogatamaDeyu FuDonovan OngEric ChenEugenie LamprechtHai PhamIsaac OngKaloyan AleksievLei LiMatthew HendersonMax BainMikel ArtetxeNishant RelanPiotr PadlewskiQi LiuRen ChenSamuel PhuaYazheng YangYi TayYuqi WangZhongkai ZhuZhihui Xie
|links| http://arxiv.org/abs/2404.12387v1 |
|updated| 2024-04-18 17:59:48 UTC |
|summary| We introduce Reka Core Flash and Edge a series of powerful multimodallanguage models trained from scratch by Reka. Reka models are able to processand reason with text images video and audio inputs. This technical reportdiscusses details of training some of these models and provides comprehensiveevaluation results. We show that Reka Edge and Reka Flash are not onlystate-of-the-art but also outperform many much larger models deliveringoutsized values for their respective compute class. Meanwhile our most capableand largest model Reka Core approaches the best frontier models on bothautomatic evaluations and blind human evaluations. On image question answeringbenchmarks e.g. MMMU VQAv2 Core performs competitively to GPT4-V.Meanwhile on multimodal chat Core ranks as the second most preferred modelunder a blind third-party human evaluation setup outperforming other modelssuch as Claude 3 Opus. On text benchmarks Core not only performs competitivelyto other frontier models on a set of well-established benchmarks e.g. MMLUGSM8K but also outperforms GPT4-0613 on human evaluation. On video questionanswering Perception-Test Core outperforms Gemini Ultra. Models are shippedin production at http://chat.reka.ai . A showcase of non cherry pickedqualitative examples can also be found at http://showcase.reka.ai . |


| Item |Content|
| --- |---|
|idx| 2404.12365v1 |
|title| When LLMs are Unfit Use FastFit: Fast and Effective Text Classification with Many Classes |
|authors| Asaf YehudaiElron Bendel
|links| http://arxiv.org/abs/2404.12365v1 |
|updated| 2024-04-18 17:48:05 UTC |
|summary| We present FastFit a method and a Python package design to provide fast andaccurate few-shot classification especially for scenarios with manysemantically similar classes. FastFit utilizes a novel approach integratingbatch contrastive learning and token-level similarity score. Compared toexisting few-shot learning packages such as SetFit Transformers or few-shotprompting of large language models via API calls FastFit significantlyimproves multiclass classification performance in speed and accuracy acrossFewMany our newly curated English benchmark and Multilingual datasets.FastFit demonstrates a 3-20x improvement in training speed completing trainingin just a few seconds. The FastFit package is now available on GitHub and PyPipresenting a user-friendly solution for NLP practitioners. |


| Item |Content|
| --- |---|
|idx| 2404.12342v1 |
|title| Large Language Models in Targeted Sentiment Analysis |
|authors| Nicolay RusnachenkoAnton GolubevNatalia Loukachevitch
|links| http://arxiv.org/abs/2404.12342v1 |
|updated| 2024-04-18 17:16:16 UTC |
|summary| In this paper we investigate the use of decoder-based generative transformersfor extracting sentiment towards the named entities in Russian news articles.We study sentiment analysis capabilities of instruction-tuned large languagemodels LLMs. We consider the dataset of RuSentNE-2023 in our study. The firstgroup of experiments was aimed at the evaluation of zero-shot capabilities ofLLMs with closed and open transparencies. The second covers the fine-tuning ofFlan-T5 using the chain-of-thought CoT three-hop reasoning frameworkTHoR. We found that the results of the zero-shot approaches are similar tothe results achieved by baseline fine-tuned encoder-based transformersBERT-base. Reasoning capabilities of the fine-tuned Flan-T5 models with THoRachieve at least 5 increment with the base-size model compared to the resultsof the zero-shot experiment. The best results of sentiment analysis onRuSentNE-2023 were achieved by fine-tuned Flan-T5-xl which surpassed theresults of previous state-of-the-art transformer-based classifiers. Our CoTapplication framework is publicly available:https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework |


| Item |Content|
| --- |---|
|idx| 2404.12318v1 |
|title| Reuse Your Rewards: Reward Model Transfer for Zero-Shot Cross-Lingual Alignment |
|authors| Zhaofeng WuAnanth BalashankarYoon KimJacob EisensteinAhmad Beirami
|links| http://arxiv.org/abs/2404.12318v1 |
|updated| 2024-04-18 16:52:36 UTC |
|summary| Aligning language models LMs based on human-annotated preference data is acrucial step in obtaining practical and performant LM-based systems. Howevermultilingual human preference data are difficult to obtain at scale making itchallenging to extend this framework to diverse languages. In this work weevaluate a simple approach for zero-shot cross-lingual alignment where areward model is trained on preference data in one source language and directlyapplied to other target languages. On summarization and open-ended dialoggeneration we show that this method is consistently successful undercomprehensive evaluation settings including human evaluation: cross-linguallyaligned models are preferred by humans over unaligned models on up to 70 ofevaluation instances. We moreover find that a different-language reward modelsometimes yields better aligned models than a same-language reward model. Wealso identify best practices when there is no language-specific data for evensupervised finetuning another component in alignment. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.12390v1 |
|title| BLINK: Multimodal Large Language Models Can See but Not Perceive |
|authors| Xingyu FuYushi HuBangzheng LiYu FengHaoyu WangXudong LinDan RothNoah A. SmithWei-Chiu MaRanjay Krishna
|links| http://arxiv.org/abs/2404.12390v1 |
|updated| 2024-04-18 17:59:54 UTC |
|summary| We introduce Blink a new benchmark for multimodal language models LLMsthat focuses on core visual perception abilities not found in otherevaluations. Most of the Blink tasks can be solved by humans within a blinke.g. relative depth estimation visual correspondence forensics detectionand multi-view reasoning. However we find these perception-demanding taskscast significant challenges for current multimodal LLMs because they resistmediation through natural language. Blink reformats 14 classic computer visiontasks into 3807 multiple-choice questions paired with single or multipleimages and visual prompting. While humans get 95.70 accuracy on average Blinkis surprisingly challenging for existing multimodal LLMs: even thebest-performing GPT-4V and Gemini achieve accuracies of 51.26 and 45.72 only13.17 and 7.63 higher than random guessing indicating that such perceptionabilities have not emerged yet in recent multimodal LLMs. Our analysis alsohighlights that specialist CV models could solve these problems much bettersuggesting potential pathways for future improvements. We believe Blink willstimulate the community to help multimodal LLMs catch up with human-levelvisual perception. |


| Item |Content|
| --- |---|
|idx| 2404.12382v1 |
|title| Lazy Diffusion Transformer for Interactive Image Editing |
|authors| Yotam NitzanZongze WuRichard ZhangEli ShechtmanDaniel Cohen-OrTaesung ParkMichaël Gharbi
|links| http://arxiv.org/abs/2404.12382v1 |
|updated| 2024-04-18 17:59:27 UTC |
|summary| We introduce a novel diffusion transformer LazyDiffusion that generatespartial image updates efficiently. Our approach targets interactive imageediting applications in which starting from a blank canvas or an image a userspecifies a sequence of localized image modifications using binary masks andtext prompts. Our generator operates in two phases. First a context encoderprocesses the current canvas and user mask to produce a compact global contexttailored to the region to generate. Second conditioned on this context adiffusion-based transformer decoder synthesizes the masked pixels in a lazyfashion i.e. it only generates the masked region. This contrasts withprevious works that either regenerate the full canvas wasting time andcomputation or confine processing to a tight rectangular crop around the maskignoring the global image context altogether. Our decoders runtime scales withthe mask size which is typically small while our encoder introducesnegligible overhead. We demonstrate that our approach is competitive withstate-of-the-art inpainting methods in terms of quality and fidelity whileproviding a 10x speedup for typical user interactions where the editing maskrepresents 10 of the image. |


| Item |Content|
| --- |---|
|idx| 2404.12378v1 |
|title| 6Img-to-3D: Few-Image Large-Scale Outdoor Driving Scene Reconstruction |
|authors| Théo GierucMarius KästingschäferSebastian BernhardMathieu Salzmann
|links| http://arxiv.org/abs/2404.12378v1 |
|updated| 2024-04-18 17:58:16 UTC |
|summary| Current 3D reconstruction techniques struggle to infer unbounded scenes froma few images faithfully. Specifically existing methods have high computationaldemands require detailed pose information and cannot reconstruct occludedregions reliably. We introduce 6Img-to-3D an efficient scalabletransformer-based encoder-renderer method for single-shot image to 3Dreconstruction. Our method outputs a 3D-consistent parameterized triplane fromonly six outward-facing input images for large-scale unbounded outdoor drivingscenarios. We take a step towards resolving existing shortcomings by combiningcontracted custom cross- and self-attention mechanisms for triplaneparameterization differentiable volume rendering scene contraction and imagefeature projection. We showcase that six surround-view vehicle images from asingle timestamp without global pose information are enough to reconstruct360circ scenes during inference time taking 395 ms. Our method allowsfor example rendering third-person images and birds-eye views. Our code isavailable at https://github.com/continental/6Img-to-3D and more examples canbe found at our website here https://6Img-to-3D.GitHub.io/. |


| Item |Content|
| --- |---|
|idx| 2404.12365v1 |
|title| When LLMs are Unfit Use FastFit: Fast and Effective Text Classification with Many Classes |
|authors| Asaf YehudaiElron Bendel
|links| http://arxiv.org/abs/2404.12365v1 |
|updated| 2024-04-18 17:48:05 UTC |
|summary| We present FastFit a method and a Python package design to provide fast andaccurate few-shot classification especially for scenarios with manysemantically similar classes. FastFit utilizes a novel approach integratingbatch contrastive learning and token-level similarity score. Compared toexisting few-shot learning packages such as SetFit Transformers or few-shotprompting of large language models via API calls FastFit significantlyimproves multiclass classification performance in speed and accuracy acrossFewMany our newly curated English benchmark and Multilingual datasets.FastFit demonstrates a 3-20x improvement in training speed completing trainingin just a few seconds. The FastFit package is now available on GitHub and PyPipresenting a user-friendly solution for NLP practitioners. |


| Item |Content|
| --- |---|
|idx| 2404.12361v1 |
|title| Learning the Domain Specific Inverse NUFFT for Accelerated Spiral MRI using Diffusion Models |
|authors| Trevor J. ChanChamith S. Rajapakse
|links| http://arxiv.org/abs/2404.12361v1 |
|updated| 2024-04-18 17:40:23 UTC |
|summary| Deep learning methods for accelerated MRI achieve state-of-the-art resultsbut largely ignore additional speedups possible with noncartesian samplingtrajectories. To address this gap we created a generative diffusionmodel-based reconstruction algorithm for multi-coil highly undersampled spiralMRI. This model uses conditioning during training as well as frequency-basedguidance to ensure consistency between images and measurements. Evaluated onretrospective data we show high quality structural similarity  0.87 inreconstructed images with ultrafast scan times 0.02 seconds for a 2D image.We use this algorithm to identify a set of optimal variable-density spiraltrajectories and show large improvements in image quality compared toconventional reconstruction using the non-uniform fast Fourier transform. Bycombining efficient spiral sampling trajectories multicoil imaging and deeplearning reconstruction these methods could enable the extremely highacceleration factors needed for real-time 3D imaging. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.12391v1 |
|title| On the Content Bias in Fréchet Video Distance |
|authors| Songwei GeAniruddha MahapatraGaurav ParmarJun-Yan ZhuJia-Bin Huang
|links| http://arxiv.org/abs/2404.12391v1 |
|updated| 2024-04-18 17:59:58 UTC |
|summary| Frechet Video Distance FVD a prominent metric for evaluating videogeneration models is known to conflict with human perception occasionally. Inthis paper we aim to explore the extent of FVDs bias toward per-frame qualityover temporal realism and identify its sources. We first quantify the FVDssensitivity to the temporal axis by decoupling the frame and motion quality andfind that the FVD increases only slightly with large temporal corruption. Wethen analyze the generated videos and show that via careful sampling from alarge set of generated videos that do not contain motions one can drasticallydecrease FVD without improving the temporal quality. Both studies suggest FVDsbias towards the quality of individual frames. We further observe that the biascan be attributed to the features extracted from a supervised video classifiertrained on the content-biased dataset. We show that FVD with features extractedfrom the recent large-scale self-supervised video models is less biased towardimage quality. Finally we revisit a few real-world examples to validate ourhypothesis. |


| Item |Content|
| --- |---|
|idx| 2404.12386v1 |
|title| SOHES: Self-supervised Open-world Hierarchical Entity Segmentation |
|authors| Shengcao CaoJiuxiang GuJason KuenHao TanRuiyi ZhangHandong ZhaoAni NenkovaLiang-Yan GuiTong SunYu-Xiong Wang
|links| http://arxiv.org/abs/2404.12386v1 |
|updated| 2024-04-18 17:59:46 UTC |
|summary| Open-world entity segmentation as an emerging computer vision task aims atsegmenting entities in images without being restricted by pre-defined classesoffering impressive generalization capabilities on unseen images and concepts.Despite its promise existing entity segmentation methods like Segment AnythingModel SAM rely heavily on costly expert annotators. This work presentsSelf-supervised Open-world Hierarchical Entity Segmentation SOHES a novelapproach that eliminates the need for human annotations. SOHES operates inthree phases: self-exploration self-instruction and self-correction. Given apre-trained self-supervised representation we produce abundant high-qualitypseudo-labels through visual feature clustering. Then we train a segmentationmodel on the pseudo-labels and rectify the noises in pseudo-labels via ateacher-student mutual-learning procedure. Beyond segmenting entities SOHESalso captures their constituent parts providing a hierarchical understandingof visual entities. Using raw images as the sole training data our methodachieves unprecedented performance in self-supervised open-world segmentationmarking a significant milestone towards high-quality open-world entitysegmentation in the absence of human-annotated masks. Project page:https://SOHES.github.io. |


| Item |Content|
| --- |---|
|idx| 2404.12378v1 |
|title| 6Img-to-3D: Few-Image Large-Scale Outdoor Driving Scene Reconstruction |
|authors| Théo GierucMarius KästingschäferSebastian BernhardMathieu Salzmann
|links| http://arxiv.org/abs/2404.12378v1 |
|updated| 2024-04-18 17:58:16 UTC |
|summary| Current 3D reconstruction techniques struggle to infer unbounded scenes froma few images faithfully. Specifically existing methods have high computationaldemands require detailed pose information and cannot reconstruct occludedregions reliably. We introduce 6Img-to-3D an efficient scalabletransformer-based encoder-renderer method for single-shot image to 3Dreconstruction. Our method outputs a 3D-consistent parameterized triplane fromonly six outward-facing input images for large-scale unbounded outdoor drivingscenarios. We take a step towards resolving existing shortcomings by combiningcontracted custom cross- and self-attention mechanisms for triplaneparameterization differentiable volume rendering scene contraction and imagefeature projection. We showcase that six surround-view vehicle images from asingle timestamp without global pose information are enough to reconstruct360circ scenes during inference time taking 395 ms. Our method allowsfor example rendering third-person images and birds-eye views. Our code isavailable at https://github.com/continental/6Img-to-3D and more examples canbe found at our website here https://6Img-to-3D.GitHub.io/. |


| Item |Content|
| --- |---|
|idx| 2404.12376v1 |
|title| Matching the Statistical Query Lower Bound for k-sparse Parity Problems with Stochastic Gradient Descent |
|authors| Yiwen KouZixiang ChenQuanquan GuSham M. Kakade
|links| http://arxiv.org/abs/2404.12376v1 |
|updated| 2024-04-18 17:57:53 UTC |
|summary| The k-parity problem is a classical problem in computational complexity andalgorithmic theory serving as a key benchmark for understanding computationalclasses. In this paper we solve the k-parity problem with stochasticgradient descent SGD on two-layer fully-connected neural networks. Wedemonstrate that SGD can efficiently solve the k-sparse parity problem on ad-dimensional hypercube kle Osqrtd with a sample complexity oftildeOdk-1 using 2Thetak neurons thus matching theestablished Omegadk lower bounds of Statistical Query SQ models. Ourtheoretical analysis begins by constructing a good neural network capable ofcorrectly solving the k-parity problem. We then demonstrate how a trainedneural network with SGD can effectively approximate this good network solvingthe k-parity problem with small statistical errors. Our theoretical resultsand findings are supported by empirical evidence showcasing the efficiency andefficacy of our approach. |


| Item |Content|
| --- |---|
|idx| 2404.12369v1 |
|title| KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning |
|authors| Marco ArazziSerena NicolazzoAntonino Nocera
|links| http://arxiv.org/abs/2404.12369v1 |
|updated| 2024-04-18 17:51:02 UTC |
|summary| Vertical Federated Learning VFL is a category of Federated Learning inwhich models are trained collaboratively among parties with verticallypartitioned data. Typically in a VFL scenario the labels of the samples arekept private from all the parties except for the aggregating server that isthe label owner. Nevertheless recent works discovered that by exploitinggradient information returned by the server to bottom models with theknowledge of only a small set of auxiliary labels on a very limited subset oftraining data points an adversary can infer the private labels. These attacksare known as label inference attacks in VFL. In our work we propose a novelframework called KDk that combines Knowledge Distillation and k-anonymity toprovide a defense mechanism against potential label inference attacks in a VFLscenario. Through an exhaustive experimental campaign we demonstrate that byapplying our approach the performance of the analyzed label inference attacksdecreases consistently even by more than 60 maintaining the accuracy of thewhole VFL almost unaltered. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.12391v1 |
|title| On the Content Bias in Fréchet Video Distance |
|authors| Songwei GeAniruddha MahapatraGaurav ParmarJun-Yan ZhuJia-Bin Huang
|links| http://arxiv.org/abs/2404.12391v1 |
|updated| 2024-04-18 17:59:58 UTC |
|summary| Frechet Video Distance FVD a prominent metric for evaluating videogeneration models is known to conflict with human perception occasionally. Inthis paper we aim to explore the extent of FVDs bias toward per-frame qualityover temporal realism and identify its sources. We first quantify the FVDssensitivity to the temporal axis by decoupling the frame and motion quality andfind that the FVD increases only slightly with large temporal corruption. Wethen analyze the generated videos and show that via careful sampling from alarge set of generated videos that do not contain motions one can drasticallydecrease FVD without improving the temporal quality. Both studies suggest FVDsbias towards the quality of individual frames. We further observe that the biascan be attributed to the features extracted from a supervised video classifiertrained on the content-biased dataset. We show that FVD with features extractedfrom the recent large-scale self-supervised video models is less biased towardimage quality. Finally we revisit a few real-world examples to validate ourhypothesis. |


| Item |Content|
| --- |---|
|idx| 2404.12390v1 |
|title| BLINK: Multimodal Large Language Models Can See but Not Perceive |
|authors| Xingyu FuYushi HuBangzheng LiYu FengHaoyu WangXudong LinDan RothNoah A. SmithWei-Chiu MaRanjay Krishna
|links| http://arxiv.org/abs/2404.12390v1 |
|updated| 2024-04-18 17:59:54 UTC |
|summary| We introduce Blink a new benchmark for multimodal language models LLMsthat focuses on core visual perception abilities not found in otherevaluations. Most of the Blink tasks can be solved by humans within a blinke.g. relative depth estimation visual correspondence forensics detectionand multi-view reasoning. However we find these perception-demanding taskscast significant challenges for current multimodal LLMs because they resistmediation through natural language. Blink reformats 14 classic computer visiontasks into 3807 multiple-choice questions paired with single or multipleimages and visual prompting. While humans get 95.70 accuracy on average Blinkis surprisingly challenging for existing multimodal LLMs: even thebest-performing GPT-4V and Gemini achieve accuracies of 51.26 and 45.72 only13.17 and 7.63 higher than random guessing indicating that such perceptionabilities have not emerged yet in recent multimodal LLMs. Our analysis alsohighlights that specialist CV models could solve these problems much bettersuggesting potential pathways for future improvements. We believe Blink willstimulate the community to help multimodal LLMs catch up with human-levelvisual perception. |


| Item |Content|
| --- |---|
|idx| 2404.12388v1 |
|title| VideoGigaGAN: Towards Detail-rich Video Super-Resolution |
|authors| Yiran XuTaesung ParkRichard ZhangYang ZhouEli ShechtmanFeng LiuJia-Bin HuangDifan Liu
|links| http://arxiv.org/abs/2404.12388v1 |
|updated| 2024-04-18 17:59:53 UTC |
|summary| Video super-resolution VSR approaches have shown impressive temporalconsistency in upsampled videos. However these approaches tend to generateblurrier results than their image counterparts as they are limited in theirgenerative capability. This raises a fundamental question: can we extend thesuccess of a generative image upsampler to the VSR task while preserving thetemporal consistency We introduce VideoGigaGAN a new generative VSR modelthat can produce videos with high-frequency details and temporal consistency.VideoGigaGAN builds upon a large-scale image upsampler -- GigaGAN. Simplyinflating GigaGAN to a video model by adding temporal modules produces severetemporal flickering. We identify several key issues and propose techniques thatsignificantly improve the temporal consistency of upsampled videos. Ourexperiments show that unlike previous VSR methods VideoGigaGAN generatestemporally consistent videos with more fine-grained appearance details. Wevalidate the effectiveness of VideoGigaGAN by comparing it withstate-of-the-art VSR models on public datasets and showcasing video resultswith 8times super-resolution. |


| Item |Content|
| --- |---|
|idx| 2404.12389v1 |
|title| Moving Object Segmentation: All You Need Is SAM (and Flow) |
|authors| Junyu XieCharig YangWeidi XieAndrew Zisserman
|links| http://arxiv.org/abs/2404.12389v1 |
|updated| 2024-04-18 17:59:53 UTC |
|summary| The objective of this paper is motion segmentation -- discovering andsegmenting the moving objects in a video. This is a much studied area withnumerous carefuland sometimes complex approaches and training schemesincluding: self-supervised learning learning from synthetic datasetsobject-centric representations amodal representations and many more. Ourinterest in this paper is to determine if the Segment Anything model SAM cancontribute to this task. We investigate two models for combining SAM withoptical flow that harness the segmentation power of SAM with the ability offlow to discover and group moving objects. In the first model we adapt SAM totake optical flow rather than RGB as an input. In the second SAM takes RGBas an input and flow is used as a segmentation prompt. These surprisinglysimple methods without any further modifications outperform all previousapproaches by a considerable margin in both single and multi-object benchmarks.We also extend these frame-level segmentations to sequence-level segmentationsthat maintain object identity. Again this simple model outperforms previousmethods on multiple video object segmentation benchmarks. |


| Item |Content|
| --- |---|
|idx| 2404.12387v1 |
|title| Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models |
|authors| Aitor OrmazabalChe ZhengCyprien de Masson d'AutumeDani YogatamaDeyu FuDonovan OngEric ChenEugenie LamprechtHai PhamIsaac OngKaloyan AleksievLei LiMatthew HendersonMax BainMikel ArtetxeNishant RelanPiotr PadlewskiQi LiuRen ChenSamuel PhuaYazheng YangYi TayYuqi WangZhongkai ZhuZhihui Xie
|links| http://arxiv.org/abs/2404.12387v1 |
|updated| 2024-04-18 17:59:48 UTC |
|summary| We introduce Reka Core Flash and Edge a series of powerful multimodallanguage models trained from scratch by Reka. Reka models are able to processand reason with text images video and audio inputs. This technical reportdiscusses details of training some of these models and provides comprehensiveevaluation results. We show that Reka Edge and Reka Flash are not onlystate-of-the-art but also outperform many much larger models deliveringoutsized values for their respective compute class. Meanwhile our most capableand largest model Reka Core approaches the best frontier models on bothautomatic evaluations and blind human evaluations. On image question answeringbenchmarks e.g. MMMU VQAv2 Core performs competitively to GPT4-V.Meanwhile on multimodal chat Core ranks as the second most preferred modelunder a blind third-party human evaluation setup outperforming other modelssuch as Claude 3 Opus. On text benchmarks Core not only performs competitivelyto other frontier models on a set of well-established benchmarks e.g. MMLUGSM8K but also outperforms GPT4-0613 on human evaluation. On video questionanswering Perception-Test Core outperforms Gemini Ultra. Models are shippedin production at http://chat.reka.ai . A showcase of non cherry pickedqualitative examples can also be found at http://showcase.reka.ai . |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.12376v1 |
|title| Matching the Statistical Query Lower Bound for k-sparse Parity Problems with Stochastic Gradient Descent |
|authors| Yiwen KouZixiang ChenQuanquan GuSham M. Kakade
|links| http://arxiv.org/abs/2404.12376v1 |
|updated| 2024-04-18 17:57:53 UTC |
|summary| The k-parity problem is a classical problem in computational complexity andalgorithmic theory serving as a key benchmark for understanding computationalclasses. In this paper we solve the k-parity problem with stochasticgradient descent SGD on two-layer fully-connected neural networks. Wedemonstrate that SGD can efficiently solve the k-sparse parity problem on ad-dimensional hypercube kle Osqrtd with a sample complexity oftildeOdk-1 using 2Thetak neurons thus matching theestablished Omegadk lower bounds of Statistical Query SQ models. Ourtheoretical analysis begins by constructing a good neural network capable ofcorrectly solving the k-parity problem. We then demonstrate how a trainedneural network with SGD can effectively approximate this good network solvingthe k-parity problem with small statistical errors. Our theoretical resultsand findings are supported by empirical evidence showcasing the efficiency andefficacy of our approach. |


| Item |Content|
| --- |---|
|idx| 2404.12356v1 |
|title| Improving the interpretability of GNN predictions through conformal-based graph sparsification |
|authors| Pablo Sanchez-MartinKinaan Aamir KhanIsabel Valera
|links| http://arxiv.org/abs/2404.12356v1 |
|updated| 2024-04-18 17:34:47 UTC |
|summary| Graph Neural Networks GNNs have achieved state-of-the-art performance insolving graph classification tasks. However most GNN architectures aggregateinformation from all nodes and edges in a graph regardless of their relevanceto the task at hand thus hindering the interpretability of their predictions.In contrast to prior work in this paper we propose a GNN emphtrainingapproach that jointly i finds the most predictive subgraph by removing edgesand/or nodes -- -emphwithout making assumptions about the subgraph structure-- while ii optimizing the performance of the graph classification task. Tothat end we rely on reinforcement learning to solve the resulting bi-leveloptimization with a reward function based on conformal predictions to accountfor the current in-training uncertainty of the classifier. Our empiricalresults on nine different graph classification datasets show that our methodcompetes in performance with baselines while relying on significantly sparsersubgraphs leading to more interpretable GNN-based predictions. |


| Item |Content|
| --- |---|
|idx| 2404.12312v1 |
|title| A Mean-Field Analysis of Neural Gradient Descent-Ascent: Applications to Functional Conditional Moment Equations |
|authors| Yuchen ZhuYufeng ZhangZhaoran WangZhuoran YangXiaohong Chen
|links| http://arxiv.org/abs/2404.12312v1 |
|updated| 2024-04-18 16:46:08 UTC |
|summary| We study minimax optimization problems defined over infinite-dimensionalfunction classes. In particular we restrict the functions to the class ofoverparameterized two-layer neural networks and study i the convergence ofthe gradient descent-ascent algorithm and ii the representation learning ofthe neural network. As an initial step we consider the minimax optimizationproblem stemming from estimating a functional equation defined by conditionalexpectations via adversarial estimation where the objective function isquadratic in the functional space. For this problem we establish convergenceunder the mean-field regime by considering the continuous-time andinfinite-width limit of the optimization dynamics. Under this regime gradientdescent-ascent corresponds to a Wasserstein gradient flow over the space ofprobability measures defined over the space of neural network parameters. Weprove that the Wasserstein gradient flow converges globally to a stationarypoint of the minimax objective at a mathcalOT-1  alpha-1  sublinear rate and additionally finds the solution to the functional equationwhen the regularizer of the minimax objective is strongly convex. Here Tdenotes the time and alpha is a scaling parameter of the neural network. Interms of representation learning our results show that the featurerepresentation induced by the neural networks is allowed to deviate from theinitial one by the magnitude of mathcalOalpha-1 measured in termsof the Wasserstein distance. Finally we apply our general results to concreteexamples including policy evaluation nonparametric instrumental variableregression and asset pricing. |


| Item |Content|
| --- |---|
|idx| 2404.12294v1 |
|title| floZ: Evidence estimation from posterior samples with normalizing flows |
|authors| Rahul SrinivasanMarco CrisostomiRoberto TrottaEnrico BarausseMatteo Breschi
|links| http://arxiv.org/abs/2404.12294v1 |
|updated| 2024-04-18 16:16:02 UTC |
|summary| We propose a novel method floZ based on normalizing flows for estimatingthe Bayesian evidence and its numerical uncertainty from a set of samplesdrawn from the unnormalized posterior distribution. We validate it ondistributions whose evidence is known analytically up to 15 parameter spacedimensions and compare with two state-of-the-art techniques for estimating theevidence: nested sampling which computes the evidence as its main target anda k-nearest-neighbors technique that produces evidence estimates from posteriorsamples. Provided representative samples from the target posterior areavailable our method is more robust to posterior distributions with sharpfeatures especially in higher dimensions. It has wide applicability e.g. toestimate the evidence from variational inference Markov-chain Monte Carlosamples or any other method that delivers samples from the unnormalizedposterior density. |


| Item |Content|
| --- |---|
|idx| 2404.12290v1 |
|title| Debiased Distribution Compression |
|authors| Lingxiao LiRaaz DwivediLester Mackey
|links| http://arxiv.org/abs/2404.12290v1 |
|updated| 2024-04-18 16:11:16 UTC |
|summary| Modern compression methods can summarize a target distribution mathbbPmore succinctly than i.i.d. sampling but require access to a low-bias inputsequence like a Markov chain converging quickly to mathbbP. We introduce anew suite of compression methods suitable for compression with biased inputsequences. Given n points targeting the wrong distribution and quadratictime Stein Kernel Thinning SKT returns sqrtn equal-weighted points withwidetildeOn-1/2 maximum mean discrepancy MMD to mathbb P. Forlarger-scale compression tasks Low-rank SKT achieves the same feat insub-quadratic time using an adaptive low-rank debiasing procedure that may beof independent interest. For downstream tasks that support simplex orconstant-preserving weights Stein Recombination and Stein Cholesky achieveeven greater parsimony matching the guarantees of SKT with as few asoperatornamepoly-logn weighted points. Underlying these advances are newguarantees for the quality of simplex-weighted coresets the spectral decay ofkernel matrices and the covering numbers of Stein kernel Hilbert spaces. Inour experiments our techniques provide succinct and accurate posteriorsummaries while overcoming biases due to burn-in approximate Markov chainMonte Carlo and tempering. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.12349v1 |
|title| Evaluating AI for Law: Bridging the Gap with Open-Source Solutions |
|authors| Rohan BhambhoriaSamuel DahanJonathan LiXiaodan Zhu
|links| http://arxiv.org/abs/2404.12349v1 |
|updated| 2024-04-18 17:26:01 UTC |
|summary| This study evaluates the performance of general-purpose AI like ChatGPT inlegal question-answering tasks highlighting significant risks to legalprofessionals and clients. It suggests leveraging foundational models enhancedby domain-specific knowledge to overcome these issues. The paper advocates forcreating open-source legal AI systems to improve accuracy transparency andnarrative diversity addressing general AIs shortcomings in legal contexts. |


| Item |Content|
| --- |---|
|idx| 2404.12317v1 |
|title| Large Language Models for Synthetic Participatory Planning of Shared Automated Electric Mobility Systems |
|authors| Jiangbo Yu
|links| http://arxiv.org/abs/2404.12317v1 |
|updated| 2024-04-18 16:51:23 UTC |
|summary| Unleashing the synergies of rapidly evolving mobility technologies in amulti-stakeholder landscape presents unique challenges and opportunities foraddressing urban transportation problems. This paper introduces a novelsynthetic participatory method critically leveraging large language modelsLLMs to create digital avatars representing diverse stakeholders to planshared automated electric mobility systems SAEMS. These calibratable agentscollaboratively identify objectives envision and evaluate SAEMS alternativesand strategize implementation under risks and constraints. The results of aMontreal case study indicate that a structured and parameterized workflowprovides outputs with high controllability and comprehensiveness on an SAEMSplan than generated using a single LLM-enabled expert agent. Consequently theapproach provides a promising avenue for cost-efficiently improving theinclusivity and interpretability of multi-objective transportation planningsuggesting a paradigm shift in how we envision and strategize for sustainableand equitable transportation systems. |


| Item |Content|
| --- |---|
|idx| 2404.12272v1 |
|title| Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences |
|authors| Shreya ShankarJ. D. Zamfirescu-PereiraBjörn HartmannAditya G. ParameswaranIan Arawjo
|links| http://arxiv.org/abs/2404.12272v1 |
|updated| 2024-04-18 15:45:27 UTC |
|summary| Due to the cumbersome nature of human evaluation and limitations ofcode-based evaluation Large Language Models LLMs are increasingly being usedto assist humans in evaluating LLM outputs. Yet LLM-generated evaluators simplyinherit all the problems of the LLMs they evaluate requiring further humanvalidation. We present a mixed-initiative approach to validate thevalidators -- aligning LLM-generated evaluation functions be it prompts orcode with human requirements. Our interface EvalGen provides automatedassistance to users in generating evaluation criteria and implementingassertions. While generating candidate implementations Python functions LLMgrader prompts EvalGen asks humans to grade a subset of LLM outputs thisfeedback is used to select implementations that better align with user grades.A qualitative study finds overall support for EvalGen but underscores thesubjectivity and iterative process of alignment. In particular we identify aphenomenon we dub emphcriteria drift: users need criteria to grade outputsbut grading outputs helps users define criteria. What is more some criteriaappears emphdependent on the specific LLM outputs observed rather thanindependent criteria that can be defined empha priori raising seriousquestions for approaches that assume the independence of evaluation fromobservation of model outputs. We present our interface and implementationdetails a comparison of our algorithm with a baseline approach andimplications for the design of future LLM evaluation assistants. |


| Item |Content|
| --- |---|
|idx| 2404.12259v1 |
|title| Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM |
|authors| Michelle S. LamJanice TeohJames LandayJeffrey HeerMichael S. Bernstein
|links| http://dx.doi.org/10.1145/3613904.3642830 |
|updated| 2024-04-18 15:26:02 UTC |
|summary| Data analysts have long sought to turn unstructured text data into meaningfulconcepts. Though common topic modeling and clustering focus on lower-levelkeywords and require significant interpretative work. We introduce conceptinduction a computational process that instead produces high-level conceptsdefined by explicit inclusion criteria from unstructured text. For a datasetof toxic online comments where a state-of-the-art BERTopic model outputswomen power female concept induction produces high-level concepts such asCriticism of traditional gender roles and Dismissal of womens concerns. Wepresent LLooM a concept induction algorithm that leverages large languagemodels to iteratively synthesize sampled text and propose human-interpretableconcepts of increasing generality. We then instantiate LLooM in amixed-initiative text analysis tool enabling analysts to shift their attentionfrom interpreting topics to engaging in theory-driven analysis. Throughtechnical evaluations and four analysis scenarios ranging from literaturereview to content moderation we find that LLooMs concepts improve upon theprior art of topic models in terms of quality and data coverage. In expert casestudies LLooM helped researchers to uncover new insights even from familiardatasets for example by suggesting a previously unnoticed concept of attackson out-party stances in a political social media dataset. |


| Item |Content|
| --- |---|
|idx| 2404.12075v1 |
|title| E-Vote Your Conscience: Perceptions of Coercion and Vote Buying, and the Usability of Fake Credentials in Online Voting |
|authors| Louis-Henri MerinoAlaleh AzhirHaoqian ZhangSimone ColomboBernhard TellenbachVero Estrada-GaliñanesBryan Ford
|links| http://arxiv.org/abs/2404.12075v1 |
|updated| 2024-04-18 10:57:32 UTC |
|summary| Online voting is attractive for convenience and accessibility but is moresusceptible to voter coercion and vote buying than in-person voting. Onemitigation is to give voters fake voting credentials that they can yield to acoercer. Fake credentials appear identical to real ones but cast votes thatare silently omitted from the final tally. An important unanswered question ishow ordinary voters perceive such a mitigation: whether they could understandand use fake credentials and whether the coercion risks justify the costs ofmitigation. We present the first systematic study of these questions involving150 diverse individuals in Boston Massachusetts. All participants registeredand voted in a mock election: 120 were exposed to coercion resistance viafake credentials the rest forming a control group. Of the 120 participantsexposed to fake credentials 96 understood their use. 53 reported that theywould create fake credentials in a real-world voting scenario given theopportunity. 10 mistakenly voted with a fake credential however. 22 reportedeither personal experience with or direct knowledge of coercion or vote-buyingincidents. These latter participants rated the coercion-resistant systemessentially as trustworthy as in-person voting via hand-marked paper ballots.Of the 150 total participants to use the system 87 successfully created theircredentials without assistance 83 both successfully created and properly usedtheir credentials. Participants give a System Usability Scale score of 70.4which is slightly above the industrys average score of 68. Our findings appearto support the importance of the coercion problem in general and the promiseof fake credentials as a possible mitigation but user error rates remain animportant usability challenge for future work. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.12317v1 |
|title| Large Language Models for Synthetic Participatory Planning of Shared Automated Electric Mobility Systems |
|authors| Jiangbo Yu
|links| http://arxiv.org/abs/2404.12317v1 |
|updated| 2024-04-18 16:51:23 UTC |
|summary| Unleashing the synergies of rapidly evolving mobility technologies in amulti-stakeholder landscape presents unique challenges and opportunities foraddressing urban transportation problems. This paper introduces a novelsynthetic participatory method critically leveraging large language modelsLLMs to create digital avatars representing diverse stakeholders to planshared automated electric mobility systems SAEMS. These calibratable agentscollaboratively identify objectives envision and evaluate SAEMS alternativesand strategize implementation under risks and constraints. The results of aMontreal case study indicate that a structured and parameterized workflowprovides outputs with high controllability and comprehensiveness on an SAEMSplan than generated using a single LLM-enabled expert agent. Consequently theapproach provides a promising avenue for cost-efficiently improving theinclusivity and interpretability of multi-objective transportation planningsuggesting a paradigm shift in how we envision and strategize for sustainableand equitable transportation systems. |


| Item |Content|
| --- |---|
|idx| 2404.12135v1 |
|title| mABC: multi-Agent Blockchain-Inspired Collaboration for root cause analysis in micro-services architecture |
|authors| Wei ZhangHongcheng GuoJian YangYi ZhangChaoran YanZhoujin TianHangyuan JiZhoujun LiTongliang LiTieqiao ZhengChao ChenYi LiangXu ShiLiangfan ZhengBo Zhang
|links| http://arxiv.org/abs/2404.12135v1 |
|updated| 2024-04-18 12:35:39 UTC |
|summary| The escalating complexity of micro-services architecture in cloud-nativetechnologies poses significant challenges for maintaining system stability andefficiency. To conduct root cause analysis RCA and resolution of alertevents we propose a pioneering framework multi-Agent Blockchain-inspiredCollaboration for root cause analysis in micro-services architecture mABC torevolutionize the AI for IT operations AIOps domain where multiple agentsbased on the powerful large language models LLMs perform blockchain-inspiredvoting to reach a final agreement following a standardized process forprocessing tasks and queries provided by Agent Workflow. Specifically sevenspecialized agents derived from Agent Workflow each provide valuable insightstowards root cause analysis based on their expertise and the intrinsic softwareknowledge of LLMs collaborating within a decentralized chain. To avoidpotential instability issues in LLMs and fully leverage the transparent andegalitarian advantages inherent in a decentralized structure mABC adopts adecision-making process inspired by blockchain governance principles whileconsidering the contribution index and expertise index of each agent.Experimental results on the public benchmark AIOps challenge dataset and ourcreated train-ticket dataset demonstrate superior performance in accuratelyidentifying root causes and formulating effective solutions compared toprevious strong baselines. The ablation study further highlights thesignificance of each component within mABC with Agent Workflow multi-agentand blockchain-inspired voting being crucial for achieving optimal performance.mABC offers a comprehensive automated root cause analysis and resolution inmicro-services architecture and achieves a significant improvement in the AIOpsdomain compared to existing baselines |


| Item |Content|
| --- |---|
|idx| 2404.12065v1 |
|title| RAGAR, Your Falsehood RADAR: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models |
|authors| M. Abdul KhaliqP. ChangM. MaB. PflugfelderF. Miletić
|links| http://arxiv.org/abs/2404.12065v1 |
|updated| 2024-04-18 10:25:42 UTC |
|summary| The escalating challenge of misinformation particularly in the context ofpolitical discourse necessitates advanced solutions for fact-checking. Weintroduce innovative approaches to enhance the reliability and efficiency ofmultimodal fact-checking through the integration of Large Language ModelsLLMs with Retrieval-augmented Generation RAG- based advanced reasoningtechniques. This work proposes two novel methodologies Chain of RAG CoRAGand Tree of RAG ToRAG. The approaches are designed to handle multimodalclaims by reasoning the next questions that need to be answered based onprevious evidence. Our approaches improve the accuracy of veracity predictionsand the generation of explanations over the traditional fact-checking approachof sub-question generation with chain of thought veracity prediction. Byemploying multimodal LLMs adept at analyzing both text and images thisresearch advances the capability of automated systems in identifying andcountering misinformation. |


| Item |Content|
| --- |---|
|idx| 2404.11831v1 |
|title| JointPPO: Diving Deeper into the Effectiveness of PPO in Multi-Agent Reinforcement Learning |
|authors| Chenxing LiuGuizhong Liu
|links| http://arxiv.org/abs/2404.11831v1 |
|updated| 2024-04-18 01:27:02 UTC |
|summary| While Centralized Training with Decentralized Execution CTDE has become theprevailing paradigm in Multi-Agent Reinforcement Learning MARL it may not besuitable for scenarios in which agents can fully communicate and shareobservations with each other. Fully centralized methods also know asCentralized Training with Centralized Execution CTCE methods can fullyutilize observations of all the agents by treating the entire system as asingle agent. However traditional CTCE methods suffer from scalability issuesdue to the exponential growth of the joint action space. To address thesechallenges in this paper we propose JointPPO a CTCE method that uses ProximalPolicy Optimization PPO to directly optimize the joint policy of themulti-agent system. JointPPO decomposes the joint policy into conditionalprobabilities transforming the decision-making process into a sequencegeneration task. A Transformer-based joint policy network is constructedtrained with a PPO loss tailored for the joint policy. JointPPO effectivelyhandles a large joint action space and extends PPO to multi-agent setting withtheoretical clarity and conciseness. Extensive experiments on the StarCraftMulti-Agent Challenge SMAC testbed demonstrate the superiority of JointPPOover the strong baselines. Ablation experiments and analyses are conducted toexplores the factors influencing JointPPOs performance. |


| Item |Content|
| --- |---|
|idx| 2404.11354v1 |
|title| Distributed Fractional Bayesian Learning for Adaptive Optimization |
|authors| Yaqun YangJinlong LeiGuanghui WenYiguang Hong
|links| http://arxiv.org/abs/2404.11354v1 |
|updated| 2024-04-17 13:09:33 UTC |
|summary| This paper considers a distributed adaptive optimization problem where allagents only have access to their local cost functions with a common unknownparameter whereas they mean to collaboratively estimate the true parameter andfind the optimal solution over a connected network. A general mathematicalframework for such a problem has not been studied yet. We aim to providevaluable insights for addressing parameter uncertainty in distributedoptimization problems and simultaneously find the optimal solution. Thus wepropose a novel Prediction while Optimization scheme which utilizesdistributed fractional Bayesian learning through weighted averaging on thelog-beliefs to update the beliefs of unknown parameters and distributedgradient descent for renewing the estimation of the optimal solution. Thenunder suitable assumptions we prove that all agents beliefs and decisionvariables converge almost surely to the true parameter and the optimal solutionunder the true parameter respectively. We further establish a sublinearconvergence rate for the belief sequence. Finally numerical experiments areimplemented to corroborate the theoretical analysis. |


