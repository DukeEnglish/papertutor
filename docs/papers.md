# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.15388v1 |
|title| LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models |
|authors| Yuzhang ShangMu CaiBingxin XuYong Jae LeeYan Yan
|links| http://arxiv.org/abs/2403.15388v1 |
|updated| 2024-03-22 17:59:52 UTC |
|summary| Large Multimodal Models LMMs have shown significant reasoning capabilitiesby connecting a visual encoder and a large language model. LMMs typically use afixed amount of visual tokens such as the penultimate layer features in theCLIP visual encoder as the prefix content. Recent LMMs incorporate morecomplex visual inputs such as high-resolution images and videos whichincrease the number of visual tokens significantly. However due to the designof the Transformer architecture computational costs associated with thesemodels tend to increase quadratically with the number of input tokens. Totackle this problem we explore a token reduction mechanism and find similarto prior work that many visual tokens are spatially redundant. Based on thiswe propose PruMerge a novel adaptive visual token reduction approach whichlargely reduces the number of visual tokens while maintaining comparable modelperformance. We first select the unpruned visual tokens based on theirsimilarity to class tokens and spatial tokens. We then cluster the prunedtokens based on key similarity and merge the clustered tokens with the unprunedtokens to supplement their information. Empirically when applied to LLaVA-1.5our approach can compress the visual tokens by 14.4 times on average andachieve comparable performance across diverse visual question-answering andreasoning tasks. Code and checkpoints are at https://llava-prumerge.github.io/. |


| Item |Content|
| --- |---|
|idx| 2403.15371v1 |
|title| Can large language models explore in-context? |
|authors| Akshay KrishnamurthyKeegan HarrisDylan J. FosterCyril ZhangAleksandrs Slivkins
|links| http://arxiv.org/abs/2403.15371v1 |
|updated| 2024-03-22 17:50:43 UTC |
|summary| We investigate the extent to which contemporary Large Language Models LLMscan engage in exploration a core capability in reinforcement learning anddecision making. We focus on native performance of existing LLMs withouttraining interventions. We deploy LLMs as agents in simple multi-armed banditenvironments specifying the environment description and interaction historyentirely in-context i.e. within the LLM prompt. We experiment with GPT-3.5GPT-4 and Llama2 using a variety of prompt designs and find that the modelsdo not robustly engage in exploration without substantial interventions: iAcross all of our experiments only one configuration resulted in satisfactoryexploratory behavior: GPT-4 with chain-of-thought reasoning and an externallysummarized interaction history presented as sufficient statistics ii Allother configurations did not result in robust exploratory behavior includingthose with chain-of-thought reasoning but unsummarized history. Although thesefindings can be interpreted positively they suggest that externalsummarization -- which may not be possible in more complex settings -- isimportant for obtaining desirable behavior from LLM agents. We conclude thatnon-trivial algorithmic interventions such as fine-tuning or dataset curationmay be required to empower LLM-based decision making agents in complexsettings. |


| Item |Content|
| --- |---|
|idx| 2403.15365v1 |
|title| A Transfer Attack to Image Watermarks |
|authors| Yuepeng HuZhengyuan JiangMoyang GuoNeil Gong
|links| http://arxiv.org/abs/2403.15365v1 |
|updated| 2024-03-22 17:33:11 UTC |
|summary| Watermark has been widely deployed by industry to detect AI-generated images.The robustness of such watermark-based detector against evasion attacks in thewhite-box and black-box settings is well understood in the literature. Howeverthe robustness in the no-box setting is much less understood. In particularmultiple studies claimed that image watermark is robust in such setting. Inthis work we propose a new transfer evasion attack to image watermark in theno-box setting. Our transfer attack adds a perturbation to a watermarked imageto evade multiple surrogate watermarking models trained by the attacker itselfand the perturbed watermarked image also evades the target watermarking model.Our major contribution is to show that both theoretically and empiricallywatermark-based AI-generated image detector is not robust to evasion attackseven if the attacker does not have access to the watermarking model nor thedetection API. |


| Item |Content|
| --- |---|
|idx| 2403.15364v1 |
|title| Towards Knowledge-Grounded Natural Language Understanding and Generation |
|authors| Chenxi Whitehouse
|links| http://arxiv.org/abs/2403.15364v1 |
|updated| 2024-03-22 17:32:43 UTC |
|summary| This thesis investigates how natural language understanding and generationwith transformer models can benefit from grounding the models with knowledgerepresentations and addresses the following key research questions: i Canknowledge of entities extend its benefits beyond entity-centric tasks such asentity linking ii How can we faithfully and effectively extract suchstructured knowledge from raw text especially noisy web text iii How doother types of knowledge beyond structured knowledge contribute to improvingNLP tasks  Studies in this thesis find that incorporating relevant and up-to-dateknowledge of entities benefits fake news detection and entity-focusedcode-switching significantly enhances zero-shot cross-lingual transfer onentity-centric tasks. In terms of effective and faithful approaches toextracting structured knowledge it is observed that integrating negativeexamples and training with entity planning significantly improves performance.Additionally it is established that other general forms of knowledge such asparametric and distilled knowledge enhance multimodal and multilingualknowledge-intensive tasks. This research shows the tangible benefits of diverseknowledge integration and motivates further exploration in this direction. |


| Item |Content|
| --- |---|
|idx| 2403.15362v1 |
|title| CoLLEGe: Concept Embedding Generation for Large Language Models |
|authors| Ryan TeehanBrenden LakeMengye Ren
|links| http://arxiv.org/abs/2403.15362v1 |
|updated| 2024-03-22 17:26:05 UTC |
|summary| Current language models are unable to quickly learn new concepts on the flyoften requiring a more involved finetuning process to learn robustly. Promptingin-context is not robust to context distractions and often fails to confermuch information about the new concepts. Classic methods for few-shot wordlearning in NLP relying on global word vectors are less applicable to largelanguage models. In this paper we introduce a novel approach named CoLLEGeConcept Learning with Language Embedding Generation to modernize few-shotconcept learning. CoLLEGe is a meta-learning framework capable of generatingflexible embeddings for new concepts using a small number of example sentencesor definitions. Our primary meta-learning objective is simply to facilitate alanguage model to make next word predictions in forthcoming sentences makingit compatible with language model pretraining. We design a series of tasks totest new concept learning in challenging real-world scenarios including newword acquisition definition inference and verbal reasoning and demonstratethat our method succeeds in each setting without task-specific training. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.15388v1 |
|title| LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models |
|authors| Yuzhang ShangMu CaiBingxin XuYong Jae LeeYan Yan
|links| http://arxiv.org/abs/2403.15388v1 |
|updated| 2024-03-22 17:59:52 UTC |
|summary| Large Multimodal Models LMMs have shown significant reasoning capabilitiesby connecting a visual encoder and a large language model. LMMs typically use afixed amount of visual tokens such as the penultimate layer features in theCLIP visual encoder as the prefix content. Recent LMMs incorporate morecomplex visual inputs such as high-resolution images and videos whichincrease the number of visual tokens significantly. However due to the designof the Transformer architecture computational costs associated with thesemodels tend to increase quadratically with the number of input tokens. Totackle this problem we explore a token reduction mechanism and find similarto prior work that many visual tokens are spatially redundant. Based on thiswe propose PruMerge a novel adaptive visual token reduction approach whichlargely reduces the number of visual tokens while maintaining comparable modelperformance. We first select the unpruned visual tokens based on theirsimilarity to class tokens and spatial tokens. We then cluster the prunedtokens based on key similarity and merge the clustered tokens with the unprunedtokens to supplement their information. Empirically when applied to LLaVA-1.5our approach can compress the visual tokens by 14.4 times on average andachieve comparable performance across diverse visual question-answering andreasoning tasks. Code and checkpoints are at https://llava-prumerge.github.io/. |


| Item |Content|
| --- |---|
|idx| 2403.15385v1 |
|title| LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis |
|authors| Kevin XieJonathan LorraineTianshi CaoJun GaoJames LucasAntonio TorralbaSanja FidlerXiaohui Zeng
|links| http://arxiv.org/abs/2403.15385v1 |
|updated| 2024-03-22 17:59:37 UTC |
|summary| Recent text-to-3D generation approaches produce impressive 3D results butrequire time-consuming optimization that can take up to an hour per prompt.Amortized methods like ATT3D optimize multiple prompts simultaneously toimprove efficiency enabling fast text-to-3D synthesis. However they cannotcapture high-frequency geometry and texture details and struggle to scale tolarge prompt sets so they generalize poorly. We introduce LATTE3D addressingthese limitations to achieve fast high-quality generation on a significantlylarger prompt set. Key to our method is 1 building a scalable architecture and2 leveraging 3D data during optimization through 3D-aware diffusion priorsshape regularization and model initialization to achieve robustness to diverseand complex training prompts. LATTE3D amortizes both neural field and texturedsurface generation to produce highly detailed textured meshes in a singleforward pass. LATTE3D generates 3D objects in 400ms and can be furtherenhanced with fast test-time optimization. |


| Item |Content|
| --- |---|
|idx| 2403.15371v1 |
|title| Can large language models explore in-context? |
|authors| Akshay KrishnamurthyKeegan HarrisDylan J. FosterCyril ZhangAleksandrs Slivkins
|links| http://arxiv.org/abs/2403.15371v1 |
|updated| 2024-03-22 17:50:43 UTC |
|summary| We investigate the extent to which contemporary Large Language Models LLMscan engage in exploration a core capability in reinforcement learning anddecision making. We focus on native performance of existing LLMs withouttraining interventions. We deploy LLMs as agents in simple multi-armed banditenvironments specifying the environment description and interaction historyentirely in-context i.e. within the LLM prompt. We experiment with GPT-3.5GPT-4 and Llama2 using a variety of prompt designs and find that the modelsdo not robustly engage in exploration without substantial interventions: iAcross all of our experiments only one configuration resulted in satisfactoryexploratory behavior: GPT-4 with chain-of-thought reasoning and an externallysummarized interaction history presented as sufficient statistics ii Allother configurations did not result in robust exploratory behavior includingthose with chain-of-thought reasoning but unsummarized history. Although thesefindings can be interpreted positively they suggest that externalsummarization -- which may not be possible in more complex settings -- isimportant for obtaining desirable behavior from LLM agents. We conclude thatnon-trivial algorithmic interventions such as fine-tuning or dataset curationmay be required to empower LLM-based decision making agents in complexsettings. |


| Item |Content|
| --- |---|
|idx| 2403.15362v1 |
|title| CoLLEGe: Concept Embedding Generation for Large Language Models |
|authors| Ryan TeehanBrenden LakeMengye Ren
|links| http://arxiv.org/abs/2403.15362v1 |
|updated| 2024-03-22 17:26:05 UTC |
|summary| Current language models are unable to quickly learn new concepts on the flyoften requiring a more involved finetuning process to learn robustly. Promptingin-context is not robust to context distractions and often fails to confermuch information about the new concepts. Classic methods for few-shot wordlearning in NLP relying on global word vectors are less applicable to largelanguage models. In this paper we introduce a novel approach named CoLLEGeConcept Learning with Language Embedding Generation to modernize few-shotconcept learning. CoLLEGe is a meta-learning framework capable of generatingflexible embeddings for new concepts using a small number of example sentencesor definitions. Our primary meta-learning objective is simply to facilitate alanguage model to make next word predictions in forthcoming sentences makingit compatible with language model pretraining. We design a series of tasks totest new concept learning in challenging real-world scenarios including newword acquisition definition inference and verbal reasoning and demonstratethat our method succeeds in each setting without task-specific training. |


| Item |Content|
| --- |---|
|idx| 2403.15341v1 |
|title| Collaborative AI Teaming in Unknown Environments via Active Goal Deduction |
|authors| Zuyuan ZhangHanhan ZhouMahdi ImaniTaeyoung LeeTian Lan
|links| http://arxiv.org/abs/2403.15341v1 |
|updated| 2024-03-22 16:50:56 UTC |
|summary| With the advancements of artificial intelligence AI were seeing morescenarios that require AI to work closely with other agents whose goals andstrategies might not be known beforehand. However existing approaches fortraining collaborative agents often require defined and known reward signalsand cannot address the problem of teaming with unknown agents that often havelatent objectives/rewards. In response to this challenge we propose teamingwith unknown agents framework which leverages kernel density Bayesian inverselearning method for active goal deduction and utilizes pre-trainedgoal-conditioned policies to enable zero-shot policy adaptation. We prove thatunbiased reward estimates in our framework are sufficient for optimal teamingwith unknown agents. We further evaluate the framework of redesignedmulti-agent particle and StarCraft II micromanagement environments with diverseunknown agents of different behaviors/rewards. Empirical results demonstratethat our framework significantly advances the teaming performance of AI andunknown agents in a wide range of collaborative scenarios. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.15389v1 |
|title| DiffusionMTL: Learning Multi-Task Denoising Diffusion Model from Partially Annotated Data |
|authors| Hanrong YeDan Xu
|links| http://arxiv.org/abs/2403.15389v1 |
|updated| 2024-03-22 17:59:58 UTC |
|summary| Recently there has been an increased interest in the practical problem oflearning multiple dense scene understanding tasks from partially annotateddata where each training sample is only labeled for a subset of the tasks. Themissing of task labels in training leads to low-quality and noisy predictionsas can be observed from state-of-the-art methods. To tackle this issue wereformulate the partially-labeled multi-task dense prediction as a pixel-leveldenoising problem and propose a novel multi-task denoising diffusion frameworkcoined as DiffusionMTL. It designs a joint diffusion and denoising paradigm tomodel a potential noisy distribution in the task prediction or feature maps andgenerate rectified outputs for different tasks. To exploit multi-taskconsistency in denoising we further introduce a Multi-Task Conditioningstrategy which can implicitly utilize the complementary nature of the tasks tohelp learn the unlabeled tasks leading to an improvement in the denoisingperformance of the different tasks. Extensive quantitative and qualitativeexperiments demonstrate that the proposed multi-task denoising diffusion modelcan significantly improve multi-task prediction maps and outperform thestate-of-the-art methods on three challenging multi-task benchmarks under twodifferent partial-labeling evaluation settings. The code is available athttps://prismformore.github.io/diffusionmtl/. |


| Item |Content|
| --- |---|
|idx| 2403.15385v1 |
|title| LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis |
|authors| Kevin XieJonathan LorraineTianshi CaoJun GaoJames LucasAntonio TorralbaSanja FidlerXiaohui Zeng
|links| http://arxiv.org/abs/2403.15385v1 |
|updated| 2024-03-22 17:59:37 UTC |
|summary| Recent text-to-3D generation approaches produce impressive 3D results butrequire time-consuming optimization that can take up to an hour per prompt.Amortized methods like ATT3D optimize multiple prompts simultaneously toimprove efficiency enabling fast text-to-3D synthesis. However they cannotcapture high-frequency geometry and texture details and struggle to scale tolarge prompt sets so they generalize poorly. We introduce LATTE3D addressingthese limitations to achieve fast high-quality generation on a significantlylarger prompt set. Key to our method is 1 building a scalable architecture and2 leveraging 3D data during optimization through 3D-aware diffusion priorsshape regularization and model initialization to achieve robustness to diverseand complex training prompts. LATTE3D amortizes both neural field and texturedsurface generation to produce highly detailed textured meshes in a singleforward pass. LATTE3D generates 3D objects in 400ms and can be furtherenhanced with fast test-time optimization. |


| Item |Content|
| --- |---|
|idx| 2403.15371v1 |
|title| Can large language models explore in-context? |
|authors| Akshay KrishnamurthyKeegan HarrisDylan J. FosterCyril ZhangAleksandrs Slivkins
|links| http://arxiv.org/abs/2403.15371v1 |
|updated| 2024-03-22 17:50:43 UTC |
|summary| We investigate the extent to which contemporary Large Language Models LLMscan engage in exploration a core capability in reinforcement learning anddecision making. We focus on native performance of existing LLMs withouttraining interventions. We deploy LLMs as agents in simple multi-armed banditenvironments specifying the environment description and interaction historyentirely in-context i.e. within the LLM prompt. We experiment with GPT-3.5GPT-4 and Llama2 using a variety of prompt designs and find that the modelsdo not robustly engage in exploration without substantial interventions: iAcross all of our experiments only one configuration resulted in satisfactoryexploratory behavior: GPT-4 with chain-of-thought reasoning and an externallysummarized interaction history presented as sufficient statistics ii Allother configurations did not result in robust exploratory behavior includingthose with chain-of-thought reasoning but unsummarized history. Although thesefindings can be interpreted positively they suggest that externalsummarization -- which may not be possible in more complex settings -- isimportant for obtaining desirable behavior from LLM agents. We conclude thatnon-trivial algorithmic interventions such as fine-tuning or dataset curationmay be required to empower LLM-based decision making agents in complexsettings. |


| Item |Content|
| --- |---|
|idx| 2403.15370v1 |
|title| Augmented Reality based Simulated Data (ARSim) with multi-view consistency for AV perception networks |
|authors| Aqeel AnwarTae Eun ChoeZian WangSanja FidlerMinwoo Park
|links| http://arxiv.org/abs/2403.15370v1 |
|updated| 2024-03-22 17:49:11 UTC |
|summary| Detecting a diverse range of objects under various driving scenarios isessential for the effectiveness of autonomous driving systems. However thereal-world data collected often lacks the necessary diversity presenting along-tail distribution. Although synthetic data has been utilized to overcomethis issue by generating virtual scenes it faces hurdles such as a significantdomain gap and the substantial efforts required from 3D artists to createrealistic environments. To overcome these challenges we present ARSim a fullyautomated comprehensive modular framework designed to enhance real multi-viewimage data with 3D synthetic objects of interest. The proposed methodintegrates domain adaptation and randomization strategies to address covariateshift between real and simulated data by inferring essential domain attributesfrom real data and employing simulation-based randomization for otherattributes. We construct a simplified virtual scene using real data andstrategically place 3D synthetic assets within it. Illumination is achieved byestimating light distribution from multiple images capturing the surroundingsof the vehicle. Camera parameters from real data are employed to rendersynthetic assets in each frame. The resulting augmented multi-view consistentdataset is used to train a multi-camera perception network for autonomousvehicles. Experimental results on various AV perception tasks demonstrate thesuperior performance of networks trained on the augmented dataset. |


| Item |Content|
| --- |---|
|idx| 2403.15365v1 |
|title| A Transfer Attack to Image Watermarks |
|authors| Yuepeng HuZhengyuan JiangMoyang GuoNeil Gong
|links| http://arxiv.org/abs/2403.15365v1 |
|updated| 2024-03-22 17:33:11 UTC |
|summary| Watermark has been widely deployed by industry to detect AI-generated images.The robustness of such watermark-based detector against evasion attacks in thewhite-box and black-box settings is well understood in the literature. Howeverthe robustness in the no-box setting is much less understood. In particularmultiple studies claimed that image watermark is robust in such setting. Inthis work we propose a new transfer evasion attack to image watermark in theno-box setting. Our transfer attack adds a perturbation to a watermarked imageto evade multiple surrogate watermarking models trained by the attacker itselfand the perturbed watermarked image also evades the target watermarking model.Our major contribution is to show that both theoretically and empiricallywatermark-based AI-generated image detector is not robust to evasion attackseven if the attacker does not have access to the watermarking model nor thedetection API. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.15389v1 |
|title| DiffusionMTL: Learning Multi-Task Denoising Diffusion Model from Partially Annotated Data |
|authors| Hanrong YeDan Xu
|links| http://arxiv.org/abs/2403.15389v1 |
|updated| 2024-03-22 17:59:58 UTC |
|summary| Recently there has been an increased interest in the practical problem oflearning multiple dense scene understanding tasks from partially annotateddata where each training sample is only labeled for a subset of the tasks. Themissing of task labels in training leads to low-quality and noisy predictionsas can be observed from state-of-the-art methods. To tackle this issue wereformulate the partially-labeled multi-task dense prediction as a pixel-leveldenoising problem and propose a novel multi-task denoising diffusion frameworkcoined as DiffusionMTL. It designs a joint diffusion and denoising paradigm tomodel a potential noisy distribution in the task prediction or feature maps andgenerate rectified outputs for different tasks. To exploit multi-taskconsistency in denoising we further introduce a Multi-Task Conditioningstrategy which can implicitly utilize the complementary nature of the tasks tohelp learn the unlabeled tasks leading to an improvement in the denoisingperformance of the different tasks. Extensive quantitative and qualitativeexperiments demonstrate that the proposed multi-task denoising diffusion modelcan significantly improve multi-task prediction maps and outperform thestate-of-the-art methods on three challenging multi-task benchmarks under twodifferent partial-labeling evaluation settings. The code is available athttps://prismformore.github.io/diffusionmtl/. |


| Item |Content|
| --- |---|
|idx| 2403.15388v1 |
|title| LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models |
|authors| Yuzhang ShangMu CaiBingxin XuYong Jae LeeYan Yan
|links| http://arxiv.org/abs/2403.15388v1 |
|updated| 2024-03-22 17:59:52 UTC |
|summary| Large Multimodal Models LMMs have shown significant reasoning capabilitiesby connecting a visual encoder and a large language model. LMMs typically use afixed amount of visual tokens such as the penultimate layer features in theCLIP visual encoder as the prefix content. Recent LMMs incorporate morecomplex visual inputs such as high-resolution images and videos whichincrease the number of visual tokens significantly. However due to the designof the Transformer architecture computational costs associated with thesemodels tend to increase quadratically with the number of input tokens. Totackle this problem we explore a token reduction mechanism and find similarto prior work that many visual tokens are spatially redundant. Based on thiswe propose PruMerge a novel adaptive visual token reduction approach whichlargely reduces the number of visual tokens while maintaining comparable modelperformance. We first select the unpruned visual tokens based on theirsimilarity to class tokens and spatial tokens. We then cluster the prunedtokens based on key similarity and merge the clustered tokens with the unprunedtokens to supplement their information. Empirically when applied to LLaVA-1.5our approach can compress the visual tokens by 14.4 times on average andachieve comparable performance across diverse visual question-answering andreasoning tasks. Code and checkpoints are at https://llava-prumerge.github.io/. |


| Item |Content|
| --- |---|
|idx| 2403.15385v1 |
|title| LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis |
|authors| Kevin XieJonathan LorraineTianshi CaoJun GaoJames LucasAntonio TorralbaSanja FidlerXiaohui Zeng
|links| http://arxiv.org/abs/2403.15385v1 |
|updated| 2024-03-22 17:59:37 UTC |
|summary| Recent text-to-3D generation approaches produce impressive 3D results butrequire time-consuming optimization that can take up to an hour per prompt.Amortized methods like ATT3D optimize multiple prompts simultaneously toimprove efficiency enabling fast text-to-3D synthesis. However they cannotcapture high-frequency geometry and texture details and struggle to scale tolarge prompt sets so they generalize poorly. We introduce LATTE3D addressingthese limitations to achieve fast high-quality generation on a significantlylarger prompt set. Key to our method is 1 building a scalable architecture and2 leveraging 3D data during optimization through 3D-aware diffusion priorsshape regularization and model initialization to achieve robustness to diverseand complex training prompts. LATTE3D amortizes both neural field and texturedsurface generation to produce highly detailed textured meshes in a singleforward pass. LATTE3D generates 3D objects in 400ms and can be furtherenhanced with fast test-time optimization. |


| Item |Content|
| --- |---|
|idx| 2403.15383v1 |
|title| ThemeStation: Generating Theme-Aware 3D Assets from Few Exemplars |
|authors| Zhenwei WangTengfei WangGerhard HanckeZiwei LiuRynson W. H. Lau
|links| http://arxiv.org/abs/2403.15383v1 |
|updated| 2024-03-22 17:59:01 UTC |
|summary| Real-world applications often require a large gallery of 3D assets that sharea consistent theme. While remarkable advances have been made in general 3Dcontent creation from text or image synthesizing customized 3D assetsfollowing the shared theme of input 3D exemplars remains an open andchallenging problem. In this work we present ThemeStation a novel approachfor theme-aware 3D-to-3D generation. ThemeStation synthesizes customized 3Dassets based on given few exemplars with two goals: 1 unity for generating 3Dassets that thematically align with the given exemplars and 2 diversity forgenerating 3D assets with a high degree of variations. To this end we design atwo-stage framework that draws a concept image first followed by areference-informed 3D modeling stage. We propose a novel dual scoredistillation DSD loss to jointly leverage priors from both the inputexemplars and the synthesized concept image. Extensive experiments and userstudies confirm that ThemeStation surpasses prior works in producing diversetheme-aware 3D models with impressive quality. ThemeStation also enablesvarious applications such as controllable 3D-to-3D generation. |


| Item |Content|
| --- |---|
|idx| 2403.15382v1 |
|title| DragAPart: Learning a Part-Level Motion Prior for Articulated Objects |
|authors| Ruining LiChuanxia ZhengChristian RupprechtAndrea Vedaldi
|links| http://arxiv.org/abs/2403.15382v1 |
|updated| 2024-03-22 17:58:59 UTC |
|summary| We introduce DragAPart a method that given an image and a set of drags asinput can generate a new image of the same object in a new state compatiblewith the action of the drags. Differently from prior works that focused onrepositioning objects DragAPart predicts part-level interactions such asopening and closing a drawer. We study this problem as a proxy for learning ageneralist motion model not restricted to a specific kinematic structure orobject category. To this end we start from a pre-trained image generator andfine-tune it on a new synthetic dataset Drag-a-Move which we introduce.Combined with a new encoding for the drags and dataset randomization the newmodel generalizes well to real images and different categories. Compared toprior motion-controlled generators we demonstrate much better part-levelmotion understanding. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.15312v1 |
|title| A Wasserstein perspective of Vanilla GANs |
|authors| Lea KunkelMathias Trabs
|links| http://arxiv.org/abs/2403.15312v1 |
|updated| 2024-03-22 16:04:26 UTC |
|summary| The empirical success of Generative Adversarial Networks GANs caused anincreasing interest in theoretical research. The statistical literature ismainly focused on Wasserstein GANs and generalizations thereof whichespecially allow for good dimension reduction properties. Statistical resultsfor Vanilla GANs the original optimization problem are still rather limitedand require assumptions such as smooth activation functions and equaldimensions of the latent space and the ambient space. To bridge this gap wedraw a connection from Vanilla GANs to the Wasserstein distance. By doing soexisting results for Wasserstein GANs can be extended to Vanilla GANs. Inparticular we obtain an oracle inequality for Vanilla GANs in Wassersteindistance. The assumptions of this oracle inequality are designed to besatisfied by network architectures commonly used in practice such asfeedforward ReLU networks. By providing a quantitative result for theapproximation of a Lipschitz function by a feedforward ReLU network withbounded Holder norm we conclude a rate of convergence for Vanilla GANs aswell as Wasserstein GANs as estimators of the unknown probability distribution. |


| Item |Content|
| --- |---|
|idx| 2403.15263v1 |
|title| Federated Bayesian Deep Learning: The Application of Statistical Aggregation Methods to Bayesian Models |
|authors| John FischerMarko OrescaninJustin LoomisPatrick McClure
|links| http://arxiv.org/abs/2403.15263v1 |
|updated| 2024-03-22 15:02:24 UTC |
|summary| Federated learning FL is an approach to training machine learning modelsthat takes advantage of multiple distributed datasets while maintaining dataprivacy and reducing communication costs associated with sharing localdatasets. Aggregation strategies have been developed to pool or fuse theweights and biases of distributed deterministic models however moderndeterministic deep learning DL models are often poorly calibrated and lackthe ability to communicate a measure of epistemic uncertainty in predictionwhich is desirable for remote sensing platforms and safety-criticalapplications. Conversely Bayesian DL models are often well calibrated andcapable of quantifying and communicating a measure of epistemic uncertaintyalong with a competitive prediction accuracy. Unfortunately because theweights and biases in Bayesian DL models are defined by a probabilitydistribution simple application of the aggregation methods associated with FLschemes for deterministic models is either impossible or results in sub-optimalperformance. In this work we use independent and identically distributed IIDand non-IID partitions of the CIFAR-10 dataset and a fully variationalResNet-20 architecture to analyze six different aggregation strategies forBayesian DL models. Additionally we analyze the traditional federatedaveraging approach applied to an approximate Bayesian Monte Carlo dropout modelas a lightweight alternative to more complex variational inference methods inFL. We show that aggregation strategy is a key hyperparameter in the design ofa Bayesian FL system with downstream effects on accuracy calibrationuncertainty quantification training stability and client computerequirements. |


| Item |Content|
| --- |---|
|idx| 2403.15175v1 |
|title| Double Cross-fit Doubly Robust Estimators: Beyond Series Regression |
|authors| Alec McCleanSivaraman BalakrishnanEdward H. KennedyLarry Wasserman
|links| http://arxiv.org/abs/2403.15175v1 |
|updated| 2024-03-22 12:59:03 UTC |
|summary| Doubly robust estimators with cross-fitting have gained popularity in causalinference due to their favorable structure-agnostic error guarantees. Howeverwhen additional structure such as Holder smoothness is available thenmore accurate double cross-fit doubly robust DCDR estimators can beconstructed by splitting the training data and undersmoothing nuisance functionestimators on independent samples. We study a DCDR estimator of the ExpectedConditional Covariance a functional of interest in causal inference andconditional independence testing and derive a series of increasingly powerfulresults with progressively stronger assumptions. We first provide astructure-agnostic error analysis for the DCDR estimator with no assumptions onthe nuisance functions or their estimators. Then assuming the nuisancefunctions are Holder smooth but without assuming knowledge of the truesmoothness level or the covariate density we establish that DCDR estimatorswith several linear smoothers are semiparametric efficient under minimalconditions and achieve fast convergence rates in the non-sqrtn regime.When the covariate density and smoothnesses are known we propose a minimaxrate-optimal DCDR estimator based on undersmoothed kernel regression. Moreoverwe show an undersmoothed DCDR estimator satisfies a slower-than-sqrtncentral limit theorem and that inference is possible even in thenon-sqrtn regime. Finally we support our theoretical results withsimulations providing intuition for double cross-fitting and undersmoothingdemonstrating where our estimator achieves semiparametric efficiency while theusual single cross-fit estimator fails and illustrating asymptotic normalityfor the undersmoothed DCDR estimator. |


| Item |Content|
| --- |---|
|idx| 2403.15123v1 |
|title| Quantification using Permutation-Invariant Networks based on Histograms |
|authors| Olaya Pérez-MonAlejandro MoreoJuan José del CozPablo González
|links| http://arxiv.org/abs/2403.15123v1 |
|updated| 2024-03-22 11:25:38 UTC |
|summary| Quantification also known as class prevalence estimation is the supervisedlearning task in which a model is trained to predict the prevalence of eachclass in a given bag of examples. This paper investigates the application ofdeep neural networks to tasks of quantification in scenarios where it ispossible to apply a symmetric supervised approach that eliminates the need forclassification as an intermediary step directly addressing the quantificationproblem. Additionally it discusses existing permutation-invariant layersdesigned for set processing and assesses their suitability for quantification.In light of our analysis we propose HistNetQ a novel neural architecture thatrelies on a permutation-invariant representation based on histograms that isspecially suited for quantification problems. Our experiments carried out inthe only quantification competition held to date show that HistNetQoutperforms other deep neural architectures devised for set processing as wellas the state-of-the-art quantification methods. Furthermore HistNetQ offerstwo significant advantages over traditional quantification methods: i it doesnot require the labels of the training examples but only the prevalence valuesof a collection of training bags making it applicable to new scenarios andii it is able to optimize any custom quantification-oriented loss function. |


| Item |Content|
| --- |---|
|idx| 2403.15108v1 |
|title| Active Learning for Regression based on Wasserstein distance and GroupSort Neural Networks |
|authors| Benjamin BobbiaMatthias Picard
|links| http://arxiv.org/abs/2403.15108v1 |
|updated| 2024-03-22 10:51:55 UTC |
|summary| This paper addresses a new active learning strategy for regression problems.The presented Wasserstein active regression model is based on the principles ofdistribution-matching to measure the representativeness of the labeled dataset.The Wasserstein distance is computed using GroupSort Neural Networks. The useof such networks provides theoretical foundations giving a way to quantifyerrors with explicit bounds for their size and depth. This solution is combinedwith another uncertainty-based approach that is more outlier-tolerant tocomplete the query strategy. Finally this method is compared with otherclassical and recent solutions. The study empirically shows the pertinence ofsuch a representativity-uncertainty approach which provides good estimationall along the query procedure. Moreover the Wasserstein active regressionoften achieves more precise estimations and tends to improve accuracy fasterthan other models. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.15323v1 |
|title| Introduction to Human-Robot Interaction: A Multi-Perspective Introductory Course |
|authors| Tom Williams
|links| http://arxiv.org/abs/2403.15323v1 |
|updated| 2024-03-22 16:19:56 UTC |
|summary| In this paper I describe the design of an introductory course in Human-RobotInteraction. This project-driven course is designed to introduce undergraduateand graduate engineering students especially those enrolled in ComputerScience Mechanical Engineering and Robotics degree programs to key theoriesand methods used in the field of Human-Robot Interaction that they wouldotherwise be unlikely to see in those degree programs. To achieve this aim thecourse takes students all the way from stakeholder analysis to empiricalevaluation covering and integrating key Qualitative Design Computationaland Quantitative methods along the way. I detail the goals audience andformat of the course and provide a detailed walkthrough of the coursesyllabus. |


| Item |Content|
| --- |---|
|idx| 2403.15321v1 |
|title| Visual Highlighting for Situated Brushing and Linking |
|authors| Nina DoerrBenjamin LeeKatarina BaricovaDieter SchmalstiegMichael Sedlmair
|links| http://arxiv.org/abs/2403.15321v1 |
|updated| 2024-03-22 16:17:51 UTC |
|summary| Brushing and linking is widely used for visual analytics in desktopenvironments. However using this approach to link many data items betweensituated e.g. a virtual screen with data and embedded views e.g.highlighted objects in the physical environment is largely unexplored. To thisend we study the effectiveness of visual highlighting techniques in helpingusers identify and link physical referents to brushed data marks in a situatedscatterplot. In an exploratory virtual reality user study N20 we evaluatedfour highlighting techniques under different physical layouts and tasks. Wediscuss the effectiveness of these techniques as well as implications for thedesign of brushing and linking operations in situated analytics. |


| Item |Content|
| --- |---|
|idx| 2403.15285v1 |
|title| Blockchain-based Pseudonym Management for Vehicle Twin Migrations in Vehicular Edge Metaverse |
|authors| Jiawen KangXiaofeng LuoJiangtian NieTianhao WuHaibo ZhouYonghua WangDusit NiyatoShiwen MaoShengli Xie
|links| http://arxiv.org/abs/2403.15285v1 |
|updated| 2024-03-22 15:31:37 UTC |
|summary| Driven by the great advances in metaverse and edge computing technologiesvehicular edge metaverses are expected to disrupt the current paradigm ofintelligent transportation systems. As highly computerized avatars of VehicularMetaverse Users VMUs the Vehicle Twins VTs deployed in edge servers canprovide valuable metaverse services to improve driving safety and on-boardsatisfaction for their VMUs throughout journeys. To maintain uninterruptedmetaverse experiences VTs must be migrated among edge servers following themovements of vehicles. This can raise concerns about privacy breaches duringthe dynamic communications among vehicular edge metaverses. To address theseconcerns and safeguard location privacy pseudonyms as temporary identifierscan be leveraged by both VMUs and VTs to realize anonymous communications inthe physical space and virtual spaces. However existing pseudonym managementmethods fall short in meeting the extensive pseudonym demands in vehicular edgemetaverses thus dramatically diminishing the performance of privacypreservation. To this end we present a cross-metaverse empowered dualpseudonym management framework. We utilize cross-chain technology to enhancemanagement efficiency and data security for pseudonyms. Furthermore we proposea metric to assess the privacy level and employ a Multi-Agent DeepReinforcement Learning MADRL approach to obtain an optimal pseudonymgenerating strategy. Numerical results demonstrate that our proposed schemesare high-efficiency and cost-effective showcasing their promising applicationsin vehicular edge metaverses. |


| Item |Content|
| --- |---|
|idx| 2403.15216v1 |
|title| (Un)making AI Magic: a Design Taxonomy |
|authors| Maria Luce LupettiDave Murray-Rust
|links| http://dx.doi.org/10.1145/3613904.3641954 |
|updated| 2024-03-22 14:03:37 UTC |
|summary| This paper examines the role that enchantment plays in the design of AIthings by constructing a taxonomy of design approaches that increase ordecrease the perception of magic and enchantment. We start from the designdiscourse surrounding recent developments in AI technologies highlightingspecific interaction qualities such as algorithmic uncertainties and errors andarticulating relations to the rhetoric of magic and supernatural thinking.Through analyzing and reflecting upon 52 students design projects from twoeditions of a Master course in design and AI we identify seven designprinciples and unpack the effects of each in terms of enchantment anddisenchantment. We conclude by articulating ways in which this taxonomy can beapproached and appropriated by design/HCI practitioners especially to supportexploration and reflexivity. |


| Item |Content|
| --- |---|
|idx| 2403.15115v1 |
|title| Language Models in Dialogue: Conversational Maxims for Human-AI Interactions |
|authors| Erik MiehlingManish NagireddyPrasanna SattigeriElizabeth M. DalyDavid PiorkowskiJohn T. Richards
|links| http://arxiv.org/abs/2403.15115v1 |
|updated| 2024-03-22 11:16:43 UTC |
|summary| Modern language models while sophisticated exhibit some inherentshortcomings particularly in conversational settings. We claim that many ofthe observed shortcomings can be attributed to violation of one or moreconversational principles. By drawing upon extensive research from both thesocial science and AI communities we propose a set of maxims -- quantityquality relevance manner benevolence and transparency -- for describingeffective human-AI conversation. We first justify the applicability of thefirst four maxims from Grice in the context of human-AI interactions. We thenargue that two new maxims benevolence concerning the generation of andengagement with harmful content and transparency concerning recognition ofones knowledge boundaries operational constraints and intents arenecessary for addressing behavior unique to modern human-AI interactions. Theproposed maxims offer prescriptive guidance on how to assess conversationalquality between humans and LLM-driven conversational agents informing boththeir evaluation and improved design. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.15341v1 |
|title| Collaborative AI Teaming in Unknown Environments via Active Goal Deduction |
|authors| Zuyuan ZhangHanhan ZhouMahdi ImaniTaeyoung LeeTian Lan
|links| http://arxiv.org/abs/2403.15341v1 |
|updated| 2024-03-22 16:50:56 UTC |
|summary| With the advancements of artificial intelligence AI were seeing morescenarios that require AI to work closely with other agents whose goals andstrategies might not be known beforehand. However existing approaches fortraining collaborative agents often require defined and known reward signalsand cannot address the problem of teaming with unknown agents that often havelatent objectives/rewards. In response to this challenge we propose teamingwith unknown agents framework which leverages kernel density Bayesian inverselearning method for active goal deduction and utilizes pre-trainedgoal-conditioned policies to enable zero-shot policy adaptation. We prove thatunbiased reward estimates in our framework are sufficient for optimal teamingwith unknown agents. We further evaluate the framework of redesignedmulti-agent particle and StarCraft II micromanagement environments with diverseunknown agents of different behaviors/rewards. Empirical results demonstratethat our framework significantly advances the teaming performance of AI andunknown agents in a wide range of collaborative scenarios. |


| Item |Content|
| --- |---|
|idx| 2403.15137v1 |
|title| CACA Agent: Capability Collaboration based AI Agent |
|authors| Peng XuHaoran WangChuang WangXu Liu
|links| http://arxiv.org/abs/2403.15137v1 |
|updated| 2024-03-22 11:42:47 UTC |
|summary| As AI Agents based on Large Language Models LLMs have shown potential inpractical applications across various fields how to quickly deploy an AI agentand how to conveniently expand the application scenario of AI agents has becomea challenge. Previous studies mainly focused on implementing all the reasoningcapabilities of AI agents within a single LLM which often makes the model morecomplex and also reduces the extensibility of AI agent functionality. In thispaper we propose CACA Agent Capability Collaboration based AI Agent usingan open architecture inspired by service computing. CACA Agent integrates a setof collaborative capabilities to implement AI Agents not only reducing thedependence on a single LLM but also enhancing the extensibility of both theplanning abilities and the tools available to AI agents. Utilizing the proposedsystem we present a demo to illustrate the operation and the applicationscenario extension of CACA Agent. |


| Item |Content|
| --- |---|
|idx| 2403.15128v1 |
|title| An Agent-Centric Perspective on Norm Enforcement and Sanctions |
|authors| Elena YanLuis G. NardinJomi F. HübnerOlivier Boissier
|links| http://arxiv.org/abs/2403.15128v1 |
|updated| 2024-03-22 11:30:38 UTC |
|summary| In increasingly autonomous and highly distributed multi-agent systemscentralized coordination becomes impractical and raises the need for governanceand enforcement mechanisms from an agent-centric perspective. In our conceptualview sanctioning norm enforcement is part of this agent-centric approach andthey aim at promoting norm compliance while preserving agents autonomy. Thefew works dealing with sanctioning norm enforcement and sanctions from theagent-centric perspective present limitations regarding the representation ofsanctions and the comprehensiveness of their norm enforcement process. Toaddress these drawbacks we propose the NPLs an extension of the NPLnormative programming language enriched with the representation of norms andsanctions as first-class abstractions. We also propose a BDI normative agentarchitecture embedding an engine for processing the NPLs language and a setof capabilities for approaching more comprehensively the sanctioning normenforcement process. We apply our contributions in a case study for improvingthe robustness of agents decision-making in a production automation system. |


| Item |Content|
| --- |---|
|idx| 2403.14972v1 |
|title| A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning |
|authors| Changmeng ZhengDayong LiangWengyu ZhangXiao-Yong WeiTat-Seng ChuaQing Li
|links| http://arxiv.org/abs/2403.14972v1 |
|updated| 2024-03-22 06:03:07 UTC |
|summary| This paper presents a pilot study aimed at introducing multi-agent debateinto multimodal reasoning. The study addresses two key challenges: thetrivialization of opinions resulting from excessive summarization and thediversion of focus caused by distractor concepts introduced from images. Thesechallenges stem from the inductive bottom-up nature of existing debatingschemes. To address the issue we propose a deductive top-down debatingapproach called Blueprint Debate on Graphs BDoG. In BDoG debates areconfined to a blueprint graph to prevent opinion trivialization throughworld-level summarization. Moreover by storing evidence in branches within thegraph BDoG mitigates distractions caused by frequent but irrelevant concepts.Extensive experiments validate BDoG achieving state-of-the-art results inScience QA and MMBench with significant improvements over previous methods. |


| Item |Content|
| --- |---|
|idx| 2403.14879v1 |
|title| Learning to Change: Choreographing Mixed Traffic Through Lateral Control and Hierarchical Reinforcement Learning |
|authors| Dawei WangWeizi LiLei ZhuJia Pan
|links| http://arxiv.org/abs/2403.14879v1 |
|updated| 2024-03-21 23:00:10 UTC |
|summary| The management of mixed traffic that consists of robot vehicles RVs andhuman-driven vehicles HVs at complex intersections presents a multifacetedchallenge. Traditional signal controls often struggle to adapt to dynamictraffic conditions and heterogeneous vehicle types. Recent advancements haveturned to strategies based on reinforcement learning RL leveraging itsmodel-free nature real-time operation and generalizability over differentscenarios. We introduce a hierarchical RL framework to manage mixed trafficthrough precise longitudinal and lateral control of RVs. Our proposedhierarchical framework combines the state-of-the-art mixed traffic controlalgorithm as a high level decision maker to improve the performance androbustness of the whole system. Our experiments demonstrate that the frameworkcan reduce the average waiting time by up to 54 compared to thestate-of-the-art mixed traffic control method. When the RV penetration rateexceeds 60 our technique consistently outperforms conventional traffic signalcontrol programs in terms of the average waiting time for all vehicles at theintersection. |


