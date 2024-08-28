# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.15232v1 |
|title| Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations |
|authors| Yucheng JiangYijia ShaoDekun MaSina J. SemnaniMonica S. Lam
|links| http://arxiv.org/abs/2408.15232v1 |
|updated| 2024-08-27 17:50:03 UTC |
|summary| While language model LM-powered chatbots and generative search enginesexcel at answering concrete queries discovering information in the terrain ofunknown unknowns remains challenging for users. To emulate the commoneducational scenario where children/students learn by listening to andparticipating in conversations of their parents/teachers we createCollaborative STORM Co-STORM. Unlike QA systems that require users to ask allthe questions Co-STORM lets users observe and occasionally steer the discourseamong several LM agents. The agents ask questions on the users behalfallowing the user to discover unknown unknowns serendipitously. To facilitateuser interaction Co-STORM assists users in tracking the discourse byorganizing the uncovered information into a dynamic mind map ultimatelygenerating a comprehensive report as takeaways. For automatic evaluation weconstruct the WildSeek dataset by collecting real information-seeking recordswith user goals. Co-STORM outperforms baseline methods on both discourse traceand report quality. In a further human evaluation 70 of participants preferCo-STORM over a search engine and 78 favor it over a RAG chatbot. |


| Item |Content|
| --- |---|
|idx| 2408.15221v1 |
|title| LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet |
|authors| Nathaniel LiZiwen HanIan StenekerWillow PrimackRiley GoodsideHugh ZhangZifan WangCristina MenghiniSummer Yue
|links| http://arxiv.org/abs/2408.15221v1 |
|updated| 2024-08-27 17:33:30 UTC |
|summary| Recent large language model LLM defenses have greatly improved modelsability to refuse harmful queries even when adversarially attacked. HoweverLLM defenses are primarily evaluated against automated adversarial attacks in asingle turn of conversation an insufficient threat model for real-worldmalicious use. We demonstrate that multi-turn human jailbreaks uncoversignificant vulnerabilities exceeding 70 attack success rate ASR onHarmBench against defenses that report single-digit ASRs with automatedsingle-turn attacks. Human jailbreaks also reveal vulnerabilities in machineunlearning defenses successfully recovering dual-use biosecurity knowledgefrom unlearned models. We compile these results into Multi-Turn HumanJailbreaks MHJ a dataset of 2912 prompts across 537 multi-turn jailbreaks.We publicly release MHJ alongside a compendium of jailbreak tactics developedacross dozens of commercial red teaming engagements supporting researchtowards stronger LLM defenses. |


| Item |Content|
| --- |---|
|idx| 2408.15213v1 |
|title| Classifying populist language in American presidential and governor speeches using automatic text analysis |
|authors| Olaf van der VeenSemir DzeboLevi LittvayKirk HawkinsOren Dar
|links| http://arxiv.org/abs/2408.15213v1 |
|updated| 2024-08-27 17:19:57 UTC |
|summary| Populism is a concept that is often used but notoriously difficult tomeasure. Common qualitative measurements like holistic grading or contentanalysis require great amounts of time and labour making it difficult toquickly scope out which politicians should be classified as populist and whichshould not while quantitative methods show mixed results when it comes toclassifying populist rhetoric. In this paper we develop a pipeline to trainand validate an automated classification model to estimate the use of populistlanguage. We train models based on sentences that were identified as populistand pluralist in 300 US governors speeches from 2010 to 2018 and in 45speeches of presidential candidates in 2016. We find that these models classifymost speeches correctly including 84 of governor speeches and 89 ofpresidential speeches. These results extend to different time periods with 92accuracy on more recent American governors different amounts of data with asfew as 70 training sentences per category achieving similar results and whenclassifying politicians instead of individual speeches. This pipeline is thusan effective tool that can optimise the systematic and swift classification ofthe use of populist language in politicians speeches. |


| Item |Content|
| --- |---|
|idx| 2408.15204v1 |
|title| Can Unconfident LLM Annotations Be Used for Confident Conclusions? |
|authors| Kristina GligorićTijana ZrnicCinoo LeeEmmanuel J. CandèsDan Jurafsky
|links| http://arxiv.org/abs/2408.15204v1 |
|updated| 2024-08-27 17:03:18 UTC |
|summary| Large language models LLMs have shown high agreement with human ratersacross a variety of tasks demonstrating potential to ease the challenges ofhuman data collection. In computational social science CSS researchers areincreasingly leveraging LLM annotations to complement slow and expensive humanannotations. Still guidelines for collecting and using LLM annotationswithout compromising the validity of downstream conclusions remain limited. Weintroduce Confidence-Driven Inference: a method that combines LLM annotationsand LLM confidence indicators to strategically select which human annotationsshould be collected with the goal of producing accurate statistical estimatesand provably valid confidence intervals while reducing the number of humanannotations needed. Our approach comes with safeguards against LLM annotationsof poor quality guaranteeing that the conclusions will be both valid and noless accurate than if we only relied on human annotations. We demonstrate theeffectiveness of Confidence-Driven Inference over baselines in statisticalestimation tasks across three CSS settings--text politeness stance andbias--reducing the needed number of human annotations by over 25 in each.Although we use CSS settings for demonstration Confidence-Driven Inference canbe used to estimate most standard quantities across a broad range of NLPproblems. |


| Item |Content|
| --- |---|
|idx| 2408.15188v1 |
|title| Infusing Acoustic Pause Context into Text-Based Dementia Assessment |
|authors| Franziska BraunSebastian P. BayerlFlorian HönigHartmut LehfeldThomas HillemacherTobias BockletKorbinian Riedhammer
|links| http://arxiv.org/abs/2408.15188v1 |
|updated| 2024-08-27 16:44:41 UTC |
|summary| Speech pauses alongside content and structure offer a valuable andnon-invasive biomarker for detecting dementia. This work investigates the useof pause-enriched transcripts in transformer-based language models todifferentiate the cognitive states of subjects with no cognitive impairmentmild cognitive impairment and Alzheimers dementia based on their speech froma clinical assessment. We address three binary classification tasks: Onsetmonitoring and dementia exclusion. The performance is evaluated throughexperiments on a German Verbal Fluency Test and a Picture Description Testcomparing the models effectiveness across different speech productioncontexts. Starting from a textual baseline we investigate the effect ofincorporation of pause information and acoustic context. We show the testshould be chosen depending on the task and similarly lexical pauseinformation and acoustic cross-attention contribute differently. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.15237v1 |
|title| The Mamba in the Llama: Distilling and Accelerating Hybrid Models |
|authors| Junxiong WangDaniele PaliottaAvner MayAlexander M. RushTri Dao
|links| http://arxiv.org/abs/2408.15237v1 |
|updated| 2024-08-27 17:56:11 UTC |
|summary| Linear RNN architectures like Mamba can be competitive with Transformermodels in language modeling while having advantageous deploymentcharacteristics. Given the focus on training large-scale Transformer models weconsider the challenge of converting these pretrained models for deployment. Wedemonstrate that it is feasible to distill large Transformers into linear RNNsby reusing the linear projection weights from attention layers with academicGPU resources. The resulting hybrid model which incorporates a quarter of theattention layers achieves performance comparable to the original Transformerin chat benchmarks and outperforms open-source hybrid Mamba models trained fromscratch with trillions of tokens in both chat benchmarks and generalbenchmarks. Moreover we introduce a hardware-aware speculative decodingalgorithm that accelerates the inference speed of Mamba and hybrid models.Overall we show how with limited computation resources we can remove many ofthe original attention layers and generate from the resulting model moreefficiently. Our top-performing model distilled from Llama3-8B-Instructachieves a 29.61 length-controlled win rate on AlpacaEval 2 against GPT-4 and7.35 on MT-Bench surpassing the best instruction-tuned linear RNN model. |


| Item |Content|
| --- |---|
|idx| 2408.15232v1 |
|title| Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations |
|authors| Yucheng JiangYijia ShaoDekun MaSina J. SemnaniMonica S. Lam
|links| http://arxiv.org/abs/2408.15232v1 |
|updated| 2024-08-27 17:50:03 UTC |
|summary| While language model LM-powered chatbots and generative search enginesexcel at answering concrete queries discovering information in the terrain ofunknown unknowns remains challenging for users. To emulate the commoneducational scenario where children/students learn by listening to andparticipating in conversations of their parents/teachers we createCollaborative STORM Co-STORM. Unlike QA systems that require users to ask allthe questions Co-STORM lets users observe and occasionally steer the discourseamong several LM agents. The agents ask questions on the users behalfallowing the user to discover unknown unknowns serendipitously. To facilitateuser interaction Co-STORM assists users in tracking the discourse byorganizing the uncovered information into a dynamic mind map ultimatelygenerating a comprehensive report as takeaways. For automatic evaluation weconstruct the WildSeek dataset by collecting real information-seeking recordswith user goals. Co-STORM outperforms baseline methods on both discourse traceand report quality. In a further human evaluation 70 of participants preferCo-STORM over a search engine and 78 favor it over a RAG chatbot. |


| Item |Content|
| --- |---|
|idx| 2408.15217v1 |
|title| Fundus2Video: Cross-Modal Angiography Video Generation from Static Fundus Photography with Clinical Knowledge Guidance |
|authors| Weiyi ZhangSiyu HuangJiancheng YangRuoyu ChenZongyuan GeYingfeng ZhengDanli ShiMingguang He
|links| http://arxiv.org/abs/2408.15217v1 |
|updated| 2024-08-27 17:30:49 UTC |
|summary| Fundus Fluorescein Angiography FFA is a critical tool for assessing retinalvascular dynamics and aiding in the diagnosis of eye diseases. However itsinvasive nature and less accessibility compared to Color Fundus CF imagespose significant challenges. Current CF to FFA translation methods are limitedto static generation. In this work we pioneer dynamic FFA video generationfrom static CF images. We introduce an autoregressive GAN for smoothmemory-saving frame-by-frame FFA synthesis. To enhance the focus on dynamiclesion changes in FFA regions we design a knowledge mask based on clinicalexperience. Leveraging this mask our approach integrates innovative knowledgemask-guided techniques including knowledge-boosted attention knowledge-awarediscriminators and mask-enhanced patchNCE loss aimed at refining generationin critical areas and addressing the pixel misalignment challenge. Our methodachieves the best FVD of 1503.21 and PSNR of 11.81 compared to other commonvideo generation approaches. Human assessment by an ophthalmologist confirmsits high generation quality. Notably our knowledge mask surpasses supervisedlesion segmentation masks offering a promising non-invasive alternative totraditional FFA for research and clinical applications. The code is availableat https://github.com/Michi-3000/Fundus2Video. |


| Item |Content|
| --- |---|
|idx| 2408.15204v1 |
|title| Can Unconfident LLM Annotations Be Used for Confident Conclusions? |
|authors| Kristina GligorićTijana ZrnicCinoo LeeEmmanuel J. CandèsDan Jurafsky
|links| http://arxiv.org/abs/2408.15204v1 |
|updated| 2024-08-27 17:03:18 UTC |
|summary| Large language models LLMs have shown high agreement with human ratersacross a variety of tasks demonstrating potential to ease the challenges ofhuman data collection. In computational social science CSS researchers areincreasingly leveraging LLM annotations to complement slow and expensive humanannotations. Still guidelines for collecting and using LLM annotationswithout compromising the validity of downstream conclusions remain limited. Weintroduce Confidence-Driven Inference: a method that combines LLM annotationsand LLM confidence indicators to strategically select which human annotationsshould be collected with the goal of producing accurate statistical estimatesand provably valid confidence intervals while reducing the number of humanannotations needed. Our approach comes with safeguards against LLM annotationsof poor quality guaranteeing that the conclusions will be both valid and noless accurate than if we only relied on human annotations. We demonstrate theeffectiveness of Confidence-Driven Inference over baselines in statisticalestimation tasks across three CSS settings--text politeness stance andbias--reducing the needed number of human annotations by over 25 in each.Although we use CSS settings for demonstration Confidence-Driven Inference canbe used to estimate most standard quantities across a broad range of NLPproblems. |


| Item |Content|
| --- |---|
|idx| 2408.15198v1 |
|title| Automatic 8-tissue Segmentation for 6-month Infant Brains |
|authors| Yilan DongVanessa KyriakopoulouIrina GrigorescuGrainne McAlonanDafnis BatalleMaria Deprez
|links| http://arxiv.org/abs/2408.15198v1 |
|updated| 2024-08-27 16:58:23 UTC |
|summary| Numerous studies have highlighted that atypical brain developmentparticularly during infancy and toddlerhood is linked to an increasedlikelihood of being diagnosed with a neurodevelopmental condition such asautism. Accurate brain tissue segmentations for morphological analysis areessential in numerous infant studies. However due to ongoing white matter WMmyelination changing tissue contrast in T1- and T2-weighted images automatictissue segmentation in 6-month infants is particularly difficult. On the otherhand manual labelling by experts is time-consuming and labor-intensive. Inthis study we propose the first 8-tissue segmentation pipeline forsix-month-old infant brains. This pipeline utilizes domain adaptation DAtechniques to leverage our longitudinal data including neonatal imagessegmented with the neonatal Developing Human Connectome Project structuralpipeline. Our pipeline takes raw 6-month images as inputs and generates the8-tissue segmentation as outputs forming an end-to-end segmentation pipeline.The segmented tissues include WM gray matter GM cerebrospinal fluid CSFventricles cerebellum basal ganglia brainstem and hippocampus/amygdala.Cycle-Consistent Generative Adversarial Network CycleGAN and Attention U-Netwere employed to achieve the image contrast transformation between neonatal and6-month images and perform tissue segmentation on the synthesized 6-monthimages neonatal images with 6-month intensity contrast respectively.Moreover we incorporated the segmentation outputs from Infant Brain Extractionand Analysis Toolbox iBEAT and another Attention U-Net to further enhance theperformance and construct the end-to-end segmentation pipeline. Our evaluationwith real 6-month images achieved a DICE score of 0.92 an HD95 of 1.6 and anASSD of 0.42. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.15240v1 |
|title| Generative Verifiers: Reward Modeling as Next-Token Prediction |
|authors| Lunjun ZhangArian HosseiniHritik BansalMehran KazemiAviral KumarRishabh Agarwal
|links| http://arxiv.org/abs/2408.15240v1 |
|updated| 2024-08-27 17:57:45 UTC |
|summary| Verifiers or reward models are often used to enhance the reasoningperformance of large language models LLMs. A common approach is the Best-of-Nmethod where N candidate solutions generated by the LLM are ranked by averifier and the best one is selected. While LLM-based verifiers are typicallytrained as discriminative classifiers to score solutions they do not utilizethe text generation capabilities of pretrained LLMs. To overcome thislimitation we instead propose training verifiers using the ubiquitousnext-token prediction objective jointly on verification and solutiongeneration. Compared to standard verifiers such generative verifiers GenRMcan benefit from several advantages of LLMs: they integrate seamlessly withinstruction tuning enable chain-of-thought reasoning and can utilizeadditional inference-time compute via majority voting for better verification.We demonstrate that when using Gemma-based verifiers on algorithmic andgrade-school math reasoning tasks GenRM outperforms discriminative verifiersand LLM-as-a-Judge showing a 16-64 improvement in the percentage of problemssolved with Best-of-N. Furthermore we show that GenRM scales favorably acrossdataset size model capacity and inference-time compute. |


| Item |Content|
| --- |---|
|idx| 2408.15237v1 |
|title| The Mamba in the Llama: Distilling and Accelerating Hybrid Models |
|authors| Junxiong WangDaniele PaliottaAvner MayAlexander M. RushTri Dao
|links| http://arxiv.org/abs/2408.15237v1 |
|updated| 2024-08-27 17:56:11 UTC |
|summary| Linear RNN architectures like Mamba can be competitive with Transformermodels in language modeling while having advantageous deploymentcharacteristics. Given the focus on training large-scale Transformer models weconsider the challenge of converting these pretrained models for deployment. Wedemonstrate that it is feasible to distill large Transformers into linear RNNsby reusing the linear projection weights from attention layers with academicGPU resources. The resulting hybrid model which incorporates a quarter of theattention layers achieves performance comparable to the original Transformerin chat benchmarks and outperforms open-source hybrid Mamba models trained fromscratch with trillions of tokens in both chat benchmarks and generalbenchmarks. Moreover we introduce a hardware-aware speculative decodingalgorithm that accelerates the inference speed of Mamba and hybrid models.Overall we show how with limited computation resources we can remove many ofthe original attention layers and generate from the resulting model moreefficiently. Our top-performing model distilled from Llama3-8B-Instructachieves a 29.61 length-controlled win rate on AlpacaEval 2 against GPT-4 and7.35 on MT-Bench surpassing the best instruction-tuned linear RNN model. |


| Item |Content|
| --- |---|
|idx| 2408.15231v1 |
|title| DCT-CryptoNets: Scaling Private Inference in the Frequency Domain |
|authors| Arjun RoyKaushik Roy
|links| http://arxiv.org/abs/2408.15231v1 |
|updated| 2024-08-27 17:48:29 UTC |
|summary| The convergence of fully homomorphic encryption FHE and machine learningoffers unprecedented opportunities for private inference of sensitive data. FHEenables computation directly on encrypted data safeguarding the entire machinelearning pipeline including data and model confidentiality. However existingFHE-based implementations for deep neural networks face significant challengesin computational cost latency and scalability limiting their practicaldeployment. This paper introduces DCT-CryptoNets a novel approach thatleverages frequency-domain learning to tackle these issues. Our method operatesdirectly in the frequency domain utilizing the discrete cosine transform DCTcommonly employed in JPEG compression. This approach is inherently compatiblewith remote computing services where images are usually transmitted and storedin compressed formats. DCT-CryptoNets reduces the computational burden ofhomomorphic operations by focusing on perceptually relevant low-frequencycomponents. This is demonstrated by substantial latency reduction of up to5.3times compared to prior work on image classification tasks including anovel demonstration of ImageNet inference within 2.5 hours down from 12.5hours compared to prior work on equivalent compute resources. MoreoverDCT-CryptoNets improves the reliability of encrypted accuracy by reducingvariability e.g. from pm2.5 to pm1.0 on ImageNet. This studydemonstrates a promising avenue for achieving efficient and practicalprivacy-preserving deep learning on high resolution images seen in real-worldapplications. |


| Item |Content|
| --- |---|
|idx| 2408.15221v1 |
|title| LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet |
|authors| Nathaniel LiZiwen HanIan StenekerWillow PrimackRiley GoodsideHugh ZhangZifan WangCristina MenghiniSummer Yue
|links| http://arxiv.org/abs/2408.15221v1 |
|updated| 2024-08-27 17:33:30 UTC |
|summary| Recent large language model LLM defenses have greatly improved modelsability to refuse harmful queries even when adversarially attacked. HoweverLLM defenses are primarily evaluated against automated adversarial attacks in asingle turn of conversation an insufficient threat model for real-worldmalicious use. We demonstrate that multi-turn human jailbreaks uncoversignificant vulnerabilities exceeding 70 attack success rate ASR onHarmBench against defenses that report single-digit ASRs with automatedsingle-turn attacks. Human jailbreaks also reveal vulnerabilities in machineunlearning defenses successfully recovering dual-use biosecurity knowledgefrom unlearned models. We compile these results into Multi-Turn HumanJailbreaks MHJ a dataset of 2912 prompts across 537 multi-turn jailbreaks.We publicly release MHJ alongside a compendium of jailbreak tactics developedacross dozens of commercial red teaming engagements supporting researchtowards stronger LLM defenses. |


| Item |Content|
| --- |---|
|idx| 2408.15198v1 |
|title| Automatic 8-tissue Segmentation for 6-month Infant Brains |
|authors| Yilan DongVanessa KyriakopoulouIrina GrigorescuGrainne McAlonanDafnis BatalleMaria Deprez
|links| http://arxiv.org/abs/2408.15198v1 |
|updated| 2024-08-27 16:58:23 UTC |
|summary| Numerous studies have highlighted that atypical brain developmentparticularly during infancy and toddlerhood is linked to an increasedlikelihood of being diagnosed with a neurodevelopmental condition such asautism. Accurate brain tissue segmentations for morphological analysis areessential in numerous infant studies. However due to ongoing white matter WMmyelination changing tissue contrast in T1- and T2-weighted images automatictissue segmentation in 6-month infants is particularly difficult. On the otherhand manual labelling by experts is time-consuming and labor-intensive. Inthis study we propose the first 8-tissue segmentation pipeline forsix-month-old infant brains. This pipeline utilizes domain adaptation DAtechniques to leverage our longitudinal data including neonatal imagessegmented with the neonatal Developing Human Connectome Project structuralpipeline. Our pipeline takes raw 6-month images as inputs and generates the8-tissue segmentation as outputs forming an end-to-end segmentation pipeline.The segmented tissues include WM gray matter GM cerebrospinal fluid CSFventricles cerebellum basal ganglia brainstem and hippocampus/amygdala.Cycle-Consistent Generative Adversarial Network CycleGAN and Attention U-Netwere employed to achieve the image contrast transformation between neonatal and6-month images and perform tissue segmentation on the synthesized 6-monthimages neonatal images with 6-month intensity contrast respectively.Moreover we incorporated the segmentation outputs from Infant Brain Extractionand Analysis Toolbox iBEAT and another Attention U-Net to further enhance theperformance and construct the end-to-end segmentation pipeline. Our evaluationwith real 6-month images achieved a DICE score of 0.92 an HD95 of 1.6 and anASSD of 0.42. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.15242v1 |
|title| Drone-assisted Road Gaussian Splatting with Cross-view Uncertainty |
|authors| Saining ZhangBaijun YeXiaoxue ChenYuantao ChenZongzheng ZhangCheng PengYongliang ShiHao Zhao
|links| http://arxiv.org/abs/2408.15242v1 |
|updated| 2024-08-27 17:59:55 UTC |
|summary| Robust and realistic rendering for large-scale road scenes is essential inautonomous driving simulation. Recently 3D Gaussian Splatting 3D-GS has madegroundbreaking progress in neural rendering but the general fidelity oflarge-scale road scene renderings is often limited by the input imagery whichusually has a narrow field of view and focuses mainly on the street-level localarea. Intuitively the data from the drones perspective can provide acomplementary viewpoint for the data from the ground vehicles perspectiveenhancing the completeness of scene reconstruction and rendering. Howevertraining naively with aerial and ground images which exhibit large viewdisparity poses a significant convergence challenge for 3D-GS and does notdemonstrate remarkable improvements in performance on road views. In order toenhance the novel view synthesis of road views and to effectively use theaerial information we design an uncertainty-aware training method that allowsaerial images to assist in the synthesis of areas where ground images have poorlearning outcomes instead of weighting all pixels equally in 3D-GS traininglike prior work did. We are the first to introduce the cross-view uncertaintyto 3D-GS by matching the car-view ensemble-based rendering uncertainty toaerial images weighting the contribution of each pixel to the trainingprocess. Additionally to systematically quantify evaluation metrics weassemble a high-quality synthesized dataset comprising both aerial and groundimages for road scenes. |


| Item |Content|
| --- |---|
|idx| 2408.15241v1 |
|title| GenRec: Unifying Video Generation and Recognition with Diffusion Models |
|authors| Zejia WengXitong YangZhen XingZuxuan WuYu-Gang Jiang
|links| http://arxiv.org/abs/2408.15241v1 |
|updated| 2024-08-27 17:59:41 UTC |
|summary| Video diffusion models are able to generate high-quality videos by learningstrong spatial-temporal priors on large-scale datasets. In this paper we aimto investigate whether such priors derived from a generative process aresuitable for video recognition and eventually joint optimization of generationand recognition. Building upon Stable Video Diffusion we introduce GenRec thefirst unified framework trained with a random-frame conditioning process so asto learn generalized spatial-temporal representations. The resulting frameworkcan naturally supports generation and recognition and more importantly isrobust even when visual inputs contain limited information. Extensiveexperiments demonstrate the efficacy of GenRec for both recognition andgeneration. In particular GenRec achieves competitive recognition performanceoffering 75.8 and 87.2 accuracy on SSV2 and K400 respectively. GenRec alsoperforms the best class-conditioned image-to-video generation resultsachieving 46.5 and 49.3 FVD scores on SSV2 and EK-100 datasets. FurthermoreGenRec demonstrates extraordinary robustness in scenarios that only limitedframes can be observed. |


| Item |Content|
| --- |---|
|idx| 2408.15239v1 |
|title| Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation |
|authors| Xiaojuan WangBoyang ZhouBrian CurlessIra Kemelmacher-ShlizermanAleksander HolynskiSteven M. Seitz
|links| http://arxiv.org/abs/2408.15239v1 |
|updated| 2024-08-27 17:57:14 UTC |
|summary| We present a method for generating video sequences with coherent motionbetween a pair of input key frames. We adapt a pretrained large-scaleimage-to-video diffusion model originally trained to generate videos movingforward in time from a single input image for key frame interpolation i.e.to produce a video in between two input frames. We accomplish this adaptationthrough a lightweight fine-tuning technique that produces a version of themodel that instead predicts videos moving backwards in time from a single inputimage. This model along with the original forward-moving model issubsequently used in a dual-directional diffusion sampling process thatcombines the overlapping model estimates starting from each of the twokeyframes. Our experiments show that our method outperforms both existingdiffusion-based methods and traditional frame interpolation techniques. |


| Item |Content|
| --- |---|
|idx| 2408.15235v1 |
|title| Learning-based Multi-View Stereo: A Survey |
|authors| Fangjinhua WangQingtian ZhuDi ChangQuankai GaoJunlin HanTong ZhangRichard HartleyMarc Pollefeys
|links| http://arxiv.org/abs/2408.15235v1 |
|updated| 2024-08-27 17:53:18 UTC |
|summary| 3D reconstruction aims to recover the dense 3D structure of a scene. It playsan essential role in various applications such as Augmented/Virtual RealityAR/VR autonomous driving and robotics. Leveraging multiple views of a scenecaptured from different viewpoints Multi-View Stereo MVS algorithmssynthesize a comprehensive 3D representation enabling precise reconstructionin complex environments. Due to its efficiency and effectiveness MVS hasbecome a pivotal method for image-based 3D reconstruction. Recently with thesuccess of deep learning many learning-based MVS methods have been proposedachieving impressive performance against traditional methods. We categorizethese learning-based methods as: depth map-based voxel-based NeRF-based 3DGaussian Splatting-based and large feed-forward methods. Among these we focussignificantly on depth map-based methods which are the main family of MVS dueto their conciseness flexibility and scalability. In this survey we provide acomprehensive review of the literature at the time of this writing. Weinvestigate these learning-based methods summarize their performances onpopular benchmarks and discuss promising future research directions in thisarea. |


| Item |Content|
| --- |---|
|idx| 2408.15231v1 |
|title| DCT-CryptoNets: Scaling Private Inference in the Frequency Domain |
|authors| Arjun RoyKaushik Roy
|links| http://arxiv.org/abs/2408.15231v1 |
|updated| 2024-08-27 17:48:29 UTC |
|summary| The convergence of fully homomorphic encryption FHE and machine learningoffers unprecedented opportunities for private inference of sensitive data. FHEenables computation directly on encrypted data safeguarding the entire machinelearning pipeline including data and model confidentiality. However existingFHE-based implementations for deep neural networks face significant challengesin computational cost latency and scalability limiting their practicaldeployment. This paper introduces DCT-CryptoNets a novel approach thatleverages frequency-domain learning to tackle these issues. Our method operatesdirectly in the frequency domain utilizing the discrete cosine transform DCTcommonly employed in JPEG compression. This approach is inherently compatiblewith remote computing services where images are usually transmitted and storedin compressed formats. DCT-CryptoNets reduces the computational burden ofhomomorphic operations by focusing on perceptually relevant low-frequencycomponents. This is demonstrated by substantial latency reduction of up to5.3times compared to prior work on image classification tasks including anovel demonstration of ImageNet inference within 2.5 hours down from 12.5hours compared to prior work on equivalent compute resources. MoreoverDCT-CryptoNets improves the reliability of encrypted accuracy by reducingvariability e.g. from pm2.5 to pm1.0 on ImageNet. This studydemonstrates a promising avenue for achieving efficient and practicalprivacy-preserving deep learning on high resolution images seen in real-worldapplications. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.15173v1 |
|title| Exploiting Approximate Symmetry for Efficient Multi-Agent Reinforcement Learning |
|authors| Batuhan YardimNiao He
|links| http://arxiv.org/abs/2408.15173v1 |
|updated| 2024-08-27 16:11:20 UTC |
|summary| Mean-field games MFG have become significant tools for solving large-scalemulti-agent reinforcement learning problems under symmetry. However theassumption of exact symmetry limits the applicability of MFGs as real-worldscenarios often feature inherent heterogeneity. Furthermore most works on MFGassume access to a known MFG model which might not be readily available forreal-world finite-agent games. In this work we broaden the applicability ofMFGs by providing a methodology to extend any finite-player possiblyasymmetric game to an induced MFG. First we prove that N-player dynamicgames can be symmetrized and smoothly extended to the infinite-player continuumvia explicit Kirszbraun extensions. Next we propose the notion ofalphabeta-symmetric games a new class of dynamic population games thatincorporate approximate permutation invariance. For alphabeta-symmetricgames we establish explicit approximation bounds demonstrating that a Nashpolicy of the induced MFG is an approximate Nash of the N-player dynamicgame. We show that TD learning converges up to a small bias using trajectoriesof the N-player game with finite-sample guarantees permitting symmetrizedlearning without building an explicit MFG model. Finally for certain gamessatisfying monotonicity we prove a sample complexity ofwidetildemathcalOvarepsilon-6 for the N-agent game to learn anvarepsilon-Nash up to symmetrization bias. Our theory is supported byevaluations on MARL benchmarks with thousands of agents. |


| Item |Content|
| --- |---|
|idx| 2408.15136v1 |
|title| Low-Budget Simulation-Based Inference with Bayesian Neural Networks |
|authors| Arnaud DelaunoyMaxence de la Brassinne BonardeauxSiddharth Mishra-SharmaGilles Louppe
|links| http://arxiv.org/abs/2408.15136v1 |
|updated| 2024-08-27 15:19:07 UTC |
|summary| Simulation-based inference methods have been shown to be inaccurate in thedata-poor regime when training simulations are limited or expensive. Underthese circumstances the inference network is particularly prone tooverfitting and using it without accounting for the computational uncertaintyarising from the lack of identifiability of the network weights can lead tounreliable results. To address this issue we propose using Bayesian neuralnetworks in low-budget simulation-based inference thereby explicitlyaccounting for the computational uncertainty of the posterior approximation. Wedesign a family of Bayesian neural network priors that are tailored forinference and show that they lead to well-calibrated posteriors on testedbenchmarks even when as few as O10 simulations are available. This opensup the possibility of performing reliable simulation-based inference using veryexpensive simulators as we demonstrate on a problem from the field ofcosmology where single simulations are computationally expensive. We show thatBayesian neural networks produce informative and well-calibrated posteriorestimates with only a few hundred simulations. |


| Item |Content|
| --- |---|
|idx| 2408.15065v1 |
|title| The Benefits of Balance: From Information Projections to Variance Reduction |
|authors| Lang LiuRonak MehtaSoumik PalZaid Harchaoui
|links| http://arxiv.org/abs/2408.15065v1 |
|updated| 2024-08-27 13:48:15 UTC |
|summary| Data balancing across multiple modalities/sources appears in various forms inseveral foundation models e.g. CLIP and DINO achieving universalrepresentation learning. We show that this iterative algorithm usually used toavoid representation collapse enjoys an unsuspected benefit: reducing thevariance of estimators that are functionals of the empirical distribution overthese sources. We provide non-asymptotic bounds quantifying this variancereduction effect and relate them to the eigendecays of appropriately definedMarkov operators. We explain how various forms of data balancing in contrastivemultimodal learning and self-supervised clustering can be interpreted asinstances of this variance reduction scheme. |


| Item |Content|
| --- |---|
|idx| 2408.14821v1 |
|title| Data-driven Effective Modeling of Multiscale Stochastic Dynamical Systems |
|authors| Yuan ChenDongbin Xiu
|links| http://arxiv.org/abs/2408.14821v1 |
|updated| 2024-08-27 07:03:51 UTC |
|summary| We present a numerical method for learning the dynamics of slow components ofunknown multiscale stochastic dynamical systems. While the governing equationsof the systems are unknown bursts of observation data of the slow variablesare available. By utilizing the observation data our proposed method iscapable of constructing a generative stochastic model that can accuratelycapture the effective dynamics of the slow variables in distribution. Wepresent a comprehensive set of numerical examples to demonstrate theperformance of the proposed method. |


| Item |Content|
| --- |---|
|idx| 2408.14620v1 |
|title| General targeted machine learning for modern causal mediation analysis |
|authors| Richard LiuNicholas T. WilliamsKara E. RudolphIván Díaz
|links| http://arxiv.org/abs/2408.14620v1 |
|updated| 2024-08-26 20:31:26 UTC |
|summary| Causal mediation analyses investigate the mechanisms through which causesexert their effects and are therefore central to scientific progress. Theliterature on the non-parametric definition and identification of mediationaleffects in rigourous causal models has grown significantly in recent years andthere has been important progress to address challenges in the interpretationand identification of such effects. Despite great progress in the causalinference front statistical methodology for non-parametric estimation haslagged behind with few or no methods available for tackling non-parametricestimation in the presence of multiple continuous or high-dimensionalmediators. In this paper we show that the identification formulas for sixpopular non-parametric approaches to mediation analysis proposed in recentyears can be recovered from just two statistical estimands. We leverage thisfinding to propose an all-purpose one-step estimation algorithm that can becoupled with machine learning in any mediation study that uses any of these sixdefinitions of mediation. The estimators have desirable properties such assqrtn-convergence and asymptotic normality. Estimating the first-ordercorrection for the one-step estimator requires estimation of complex densityratios on the potentially high-dimensional mediators a challenge that issolved using recent advancements in so-called Riesz learning. We illustrate theproperties of our methods in a simulation study and illustrate its use on realdata to estimate the extent to which pain management practices mediate thetotal effect of having a chronic pain disorder on opioid use disorder. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.15204v1 |
|title| Can Unconfident LLM Annotations Be Used for Confident Conclusions? |
|authors| Kristina GligorićTijana ZrnicCinoo LeeEmmanuel J. CandèsDan Jurafsky
|links| http://arxiv.org/abs/2408.15204v1 |
|updated| 2024-08-27 17:03:18 UTC |
|summary| Large language models LLMs have shown high agreement with human ratersacross a variety of tasks demonstrating potential to ease the challenges ofhuman data collection. In computational social science CSS researchers areincreasingly leveraging LLM annotations to complement slow and expensive humanannotations. Still guidelines for collecting and using LLM annotationswithout compromising the validity of downstream conclusions remain limited. Weintroduce Confidence-Driven Inference: a method that combines LLM annotationsand LLM confidence indicators to strategically select which human annotationsshould be collected with the goal of producing accurate statistical estimatesand provably valid confidence intervals while reducing the number of humanannotations needed. Our approach comes with safeguards against LLM annotationsof poor quality guaranteeing that the conclusions will be both valid and noless accurate than if we only relied on human annotations. We demonstrate theeffectiveness of Confidence-Driven Inference over baselines in statisticalestimation tasks across three CSS settings--text politeness stance andbias--reducing the needed number of human annotations by over 25 in each.Although we use CSS settings for demonstration Confidence-Driven Inference canbe used to estimate most standard quantities across a broad range of NLPproblems. |


| Item |Content|
| --- |---|
|idx| 2408.15199v1 |
|title| Crossing Rays: Evaluation of Bimanual Mid-air Selection Techniques in an Immersive Environment |
|authors| DongHoon KimDongyun HanSiyeon BakIsaac Cho
|links| http://arxiv.org/abs/2408.15199v1 |
|updated| 2024-08-27 17:00:22 UTC |
|summary| Mid-air navigation offers a method of aerial travel that mitigates theconstraints associated with continuous navigation. A mid-air selectiontechnique is essential to enable such navigation. In this paper we considerfour variations of intersection-based bimanual mid-air selection techniqueswith visual aids and supporting features: Simple-Ray Simple-StripePrecision-Stripe and Cursor-Sync. We evaluate their performance and userexperience compared to an unimanual mid-air selection technique using two tasksthat require selecting a mid-air position with or without a reference object.Our findings indicate that the bimanual techniques generally demonstrate fasterselection times compared to the unimanual technique. With a supporting featurethe bimanual techniques can provide a more accurate selection than theunimanual technique. Based on our results we discuss the effect of selectiontechniques visual aids and supporting features on performance and userexperience for mid-air selection. |


| Item |Content|
| --- |---|
|idx| 2408.15177v1 |
|title| Regaining Trust: Impact of Transparent User Interface Design on Acceptance of Camera-Based In-Car Health Monitoring Systems |
|authors| Hauke SandhausMadiha Zahrah ChoksiWendy Ju
|links| http://arxiv.org/abs/2408.15177v1 |
|updated| 2024-08-27 16:21:29 UTC |
|summary| Introducing in-car health monitoring systems offers substantial potential toimprove driver safety. However camera-based sensing technologies introducesignificant privacy concerns. This study investigates the impact of transparentuser interface design on user acceptance of these systems. We conducted anonline study with 42 participants using prototypes varying in transparencychoice and deception levels. The prototypes included three onboarding designs:1 a traditional Terms and Conditions text 2 a Business Nudge design thatsubtly encouraged users to accept default data-sharing options and 3 aTransparent Walk-Through that provided clear step-by-step explanations of datause and privacy policies. Our findings indicate that transparent designsignificantly affects user experience measures including perceived creepinesstrust in data use and trustworthiness of content. Transparent onboardingprocesses enhanced user experience and trust without significantly increasingonboarding time. These findings offer practical guidance for designinguser-friendly and privacy-respecting in-car health monitoring systems. |


| Item |Content|
| --- |---|
|idx| 2408.15073v1 |
|title| Interactive dense pixel visualizations for time series and model attribution explanations |
|authors| Udo SchlegelDaniel A. Keim
|links| http://arxiv.org/abs/2408.15073v1 |
|updated| 2024-08-27 14:02:21 UTC |
|summary| The field of Explainable Artificial Intelligence XAI for Deep NeuralNetwork models has developed significantly offering numerous techniques toextract explanations from models. However evaluating explanations is often nottrivial and differences in applied metrics can be subtle especially withnon-intelligible data. Thus there is a need for visualizations tailored toexplore explanations for domains with such data e.g. time series. We proposeDAVOTS an interactive visual analytics approach to explore raw time seriesdata activations of neural networks and attributions in a dense-pixelvisualization to gain insights into the data models decisions andexplanations. To further support users in exploring large datasets we applyclustering approaches to the visualized data domains to highlight groups andpresent ordering strategies for individual and combined data exploration tofacilitate finding patterns. We visualize a CNN trained on the FordA dataset todemonstrate the approach. |


| Item |Content|
| --- |---|
|idx| 2408.15066v1 |
|title| Constraining Participation: Affordances of Feedback Features in Interfaces to Large Language Models |
|authors| Ned CooperAlexandra Zafiroglu
|links| http://arxiv.org/abs/2408.15066v1 |
|updated| 2024-08-27 13:50:37 UTC |
|summary| Large language models LLMs are now accessible to anyone with a computer aweb browser and an internet connection via browser-based interfaces shiftingthe dynamics of participation in AI development. This paper examines theaffordances of interactive feedback features in ChatGPTs interface analysinghow they shape user input and participation in LLM iteration. Drawing on asurvey of ChatGPT users and applying the mechanisms and conditions framework ofaffordances we demonstrate that these features encourage simple frequent andperformance-focused feedback while discouraging collective input anddiscussions among users. We argue that this feedback format significantlyconstrains user participation reinforcing power imbalances between users thepublic and companies developing LLMs. Our analysis contributes to the growingbody of literature on participatory AI by critically examining the limitationsof existing feedback processes and proposing directions for their redesign. Toenable more meaningful public participation in AI development we advocate fora shift away from processes focused on aligning model outputs with specificuser preferences. Instead we emphasise the need for processes that facilitatedialogue between companies and diverse publics about the purpose andapplications of LLMs. This approach requires attention to the ongoing work ofinfrastructuring - creating and sustaining the social technical andinstitutional structures necessary to address matters of concern to groupsimpacted by AI development and deployment. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.14948v1 |
|title| Decentralized Unlabeled Multi-agent Pathfinding Via Target And Priority Swapping (With Supplementary) |
|authors| Stepan DergachevKonstantin Yakovlev
|links| http://arxiv.org/abs/2408.14948v1 |
|updated| 2024-08-27 10:45:57 UTC |
|summary| In this paper we study a challenging variant of the multi-agent pathfindingproblem MAPF when a set of agents must reach a set of goal locations but itdoes not matter which agent reaches a specific goal - Anonymous MAPF AMAPF.Current optimal and suboptimal AMAPF solvers rely on the existence of acentralized controller which is in charge of both target assignment andpathfinding. We extend the state of the art and present the first AMAPF solvercapable of solving the problem at hand in a fully decentralized fashion wheneach agent makes decisions individually and relies only on the localcommunication with the others. The core of our method is a priority and targetswapping procedure tailored to produce consistent goal assignments i.e. makingsure that no two agents are heading towards the same goal. Coupled with anestablished rule-based path planning we end up with a TP-SWAP an efficientand flexible approach to solve decentralized AMAPF. On the theoretical side weprove that TP-SWAP is complete i.e. TP-SWAP guarantees that each target willbe reached by some agent. Empirically we evaluate TP-SWAP across a wide rangeof setups and compare it to both centralized and decentralized baselines.Indeed TP-SWAP outperforms the fully-decentralized competitor and can evenoutperform the semi-decentralized one i.e. the one relying on the initialconsistent goal assignment in terms of flowtime a widespread cost objectivein MAPF |


| Item |Content|
| --- |---|
|idx| 2408.14527v1 |
|title| Multi-Agent Path Finding with Real Robot Dynamics and Interdependent Tasks for Automated Warehouses |
|authors| Vassilissa Lehoux-LebacqueTomi SilanderChristelle LoiodiceSeungjoon LeeAlbert WangSofia Michel
|links| http://arxiv.org/abs/2408.14527v1 |
|updated| 2024-08-26 15:13:38 UTC |
|summary| Multi-Agent Path Finding MAPF is an important optimization problemunderlying the deployment of robots in automated warehouses and factories.Despite the large body of work on this topic most approaches make heavysimplifications both on the environment and the agents which make theresulting algorithms impractical for real-life scenarios. In this paper weconsider a realistic problem of online order delivery in a warehouse where afleet of robots bring the products belonging to each order from shelves toworkstations. This creates a stream of inter-dependent pickup and deliverytasks and the associated MAPF problem consists of computing realisticcollision-free robot trajectories fulfilling these tasks. To solve this MAPFproblem we propose an extension of the standard Prioritized Planning algorithmto deal with the inter-dependent tasks Interleaved Prioritized Planning and anovel Via-Point Star VP algorithm to compute an optimal dynamics-compliantrobot trajectory to visit a sequence of goal locations while avoiding movingobstacles. We prove the completeness of our approach and evaluate it insimulation as well as in a real warehouse. |


| Item |Content|
| --- |---|
|idx| 2408.14199v1 |
|title| A Survey on Small-Scale Testbeds for Connected and Automated Vehicles and Robot Swarms |
|authors| Armin MokhtarianJianye XuPatrick ScheffeMaximilian KloockSimon SchäferHeeseung BangViet-Anh LeSangeet UlhasJohannes BetzSean WilsonSpring BermanLiam PaullAmanda ProrokBassam Alrifaee
|links| http://dx.doi.org/10.13140/RG.2.2.16176.74248/1 |
|updated| 2024-08-26 11:54:27 UTC |
|summary| Connected and automated vehicles and robot swarms hold transformativepotential for enhancing safety efficiency and sustainability in thetransportation and manufacturing sectors. Extensive testing and validation ofthese technologies is crucial for their deployment in the real world. Whilesimulations are essential for initial testing they often have limitations incapturing the complex dynamics of real-world interactions. This limitationunderscores the importance of small-scale testbeds. These testbeds provide arealistic cost-effective and controlled environment for testing andvalidating algorithms acting as an essential intermediary between simulationand full-scale experiments. This work serves to facilitate researchers effortsin identifying existing small-scale testbeds suitable for their experiments andprovide insights for those who want to build their own. In addition itdelivers a comprehensive survey of the current landscape of these testbeds. Wederive 62 characteristics of testbeds based on the well-known sense-plan-actparadigm and offer an online table comparing 22 small-scale testbeds based onthese characteristics. The online table is hosted on our designated publicwebpage www.cpm-remote.de/testbeds and we invite testbed creators anddevelopers to contribute to it. We closely examine nine testbeds in this paperdemonstrating how the derived characteristics can be used to present testbeds.Furthermore we discuss three ongoing challenges concerning small-scaletestbeds that we identified i.e. small-scale to full-scale transitionsustainability and power and resource management. |


| Item |Content|
| --- |---|
|idx| 2408.13828v1 |
|title| Decentralized Stochastic Control in Standard Borel Spaces: Centralized MDP Reductions, Near Optimality of Finite Window Local Information, and Q-Learning |
|authors| Omar Mrani-ZentarSerdar Yüksel
|links| http://arxiv.org/abs/2408.13828v1 |
|updated| 2024-08-25 13:07:34 UTC |
|summary| Decentralized stochastic control problems are intrinsically difficult tostudy because of the inapplicability of standard tools from centralized controlsuch as dynamic programming and the resulting computational complexity. In thispaper we address some of these challenges for decentralized stochastic controlwith Borel spaces under three different but tightly related informationstructures under a unified theme: the one-step delayed information sharingpattern the K-step periodic information sharing pattern and the completelydecentralized information structure where no sharing of information occurs. Wewill show that the one-step delayed and K-step periodic problems can be reducedto a centralized MDP generalizing prior results which considered finitelinear or static models by addressing several measurability questions. Theseparated nature of policies under both information structures is thenestablished. We then provide sufficient conditions for the transition kernelsof both centralized reductions to be weak-Feller which facilitates rigorousapproximation and learning theoretic results. We will then show that for thecompletely decentralized control problem finite memory local policies are nearoptimal under a joint conditional mixing condition. This is achieved byobtaining a bound for finite memory policies which goes to zero as memory sizeincreases. We will also provide a performance bound for the K-periodic problemwhich results from replacing the full common information by a finite slidingwindow of information. The latter will depend on the condition of predictorstability in expected total variation which we will establish. We finally showthat under the periodic information sharing pattern a quantized Q-learningalgorithm converges asymptotically towards a near optimal solution. Each of theabove to our knowledge is a new contribution to the literature. |


| Item |Content|
| --- |---|
|idx| 2408.13750v1 |
|title| Multi-Agent Target Assignment and Path Finding for Intelligent Warehouse: A Cooperative Multi-Agent Deep Reinforcement Learning Perspective |
|authors| Qi LiuJianqi GaoDongjie ZhuXizheng PangPengbin ChenJingxiang GuoYanjie Li
|links| http://arxiv.org/abs/2408.13750v1 |
|updated| 2024-08-25 07:32:58 UTC |
|summary| Multi-agent target assignment and path planning TAPF are two key problemsin intelligent warehouse. However most literature only addresses one of thesetwo problems separately. In this study we propose a method to simultaneouslysolve target assignment and path planning from a perspective of cooperativemulti-agent deep reinforcement learning RL. To the best of our knowledgethis is the first work to model the TAPF problem for intelligent warehouse tocooperative multi-agent deep RL and the first to simultaneously address TAPFbased on multi-agent deep RL. Furthermore previous literature rarely considersthe physical dynamics of agents. In this study the physical dynamics of theagents is considered. Experimental results show that our method performs wellin various task settings which means that the target assignment is solvedreasonably well and the planned path is almost shortest. Moreover our methodis more time-efficient than baselines. |


