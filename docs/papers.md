# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.10210v1 |
|title| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation |
|authors| Huizhuo YuanZixiang ChenKaixuan JiQuanquan Gu
|links| http://arxiv.org/abs/2402.10210v1 |
|updated| 2024-02-15 18:59:18 UTC |
|summary| Fine-tuning Diffusion Models remains an underexplored frontier in generativeartificial intelligence GenAI especially when compared with the remarkableprogress made in fine-tuning Large Language Models LLMs. While cutting-edgediffusion models such as Stable Diffusion SD and SDXL rely on supervisedfine-tuning their performance inevitably plateaus after seeing a certainvolume of data. Recently reinforcement learning RL has been employed tofine-tune diffusion models with human preference data but it requires at leasttwo images winner and loser images for each text prompt. In this paperwe introduce an innovative technique called self-play fine-tuning for diffusionmodels SPIN-Diffusion where the diffusion model engages in competition withits earlier versions facilitating an iterative self-improvement process. Ourapproach offers an alternative to conventional supervised fine-tuning and RLstrategies significantly improving both model performance and alignment. Ourexperiments on the Pick-a-Pic dataset reveal that SPIN-Diffusion outperformsthe existing supervised fine-tuning method in aspects of human preferencealignment and visual appeal right from its first iteration. By the seconditeration it exceeds the performance of RLHF-based methods across all metricsachieving these results with less data. |


| Item |Content|
| --- |---|
|idx| 2402.10208v1 |
|title| Recovering the Pre-Fine-Tuning Weights of Generative Models |
|authors| Eliahu HorwitzJonathan KahanaYedid Hoshen
|links| http://arxiv.org/abs/2402.10208v1 |
|updated| 2024-02-15 18:59:02 UTC |
|summary| The dominant paradigm in generative modeling consists of two steps: ipre-training on a large-scale but unsafe dataset ii aligning the pre-trainedmodel with human values via fine-tuning. This practice is considered safe asno current method can recover the unsafe pre-fine-tuning model weights. Inthis paper we demonstrate that this assumption is often false. Concretely wepresent Spectral DeTuning a method that can recover the weights of thepre-fine-tuning model using a few low-rank LoRA fine-tuned models. Incontrast to previous attacks that attempt to recover pre-fine-tuningcapabilities our method aims to recover the exact pre-fine-tuning weights. Ourapproach exploits this new vulnerability against large-scale models such as apersonalized Stable Diffusion and an aligned Mistral. |


| Item |Content|
| --- |---|
|idx| 2402.10207v1 |
|title| Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment |
|authors| Rui YangXiaoman PanFeng LuoShuang QiuHan ZhongDong YuJianshu Chen
|links| http://arxiv.org/abs/2402.10207v1 |
|updated| 2024-02-15 18:58:31 UTC |
|summary| We consider the problem of multi-objective alignment of foundation modelswith human preferences which is a critical step towards helpful and harmlessAI systems. However it is generally costly and unstable to fine-tune largefoundation models using reinforcement learning RL and themulti-dimensionality heterogeneity and conflicting nature of humanpreferences further complicate the alignment process. In this paper weintroduce Rewards-in-Context RiC which conditions the response of afoundation model on multiple rewards in its prompt context and appliessupervised fine-tuning for alignment. The salient features of RiC aresimplicity and adaptivity as it only requires supervised fine-tuning of asingle foundation model and supports dynamic adjustment for user preferencesduring inference time. Inspired by the analytical solution of an abstractedconvex optimization problem our dynamic inference-time adjustment methodapproaches the Pareto-optimal solution for multiple objectives. Empiricalevidence demonstrates the efficacy of our method in aligning both LargeLanguage Models LLMs and diffusion models to accommodate diverse rewards withonly around 10 GPU hours compared with multi-objective RL baseline. |


| Item |Content|
| --- |---|
|idx| 2402.10200v1 |
|title| Chain-of-Thought Reasoning Without Prompting |
|authors| Xuezhi WangDenny Zhou
|links| http://arxiv.org/abs/2402.10200v1 |
|updated| 2024-02-15 18:55:41 UTC |
|summary| In enhancing the reasoning capabilities of large language models LLMsprior research primarily focuses on specific prompting techniques such asfew-shot or zero-shot chain-of-thought CoT prompting. These methods whileeffective often involve manually intensive prompt engineering. Our study takesa novel approach by asking: Can LLMs reason effectively without prompting Ourfindings reveal that intriguingly CoT reasoning paths can be elicited frompre-trained LLMs by simply altering the textitdecoding process. Rather thanconventional greedy decoding we investigate the top-k alternative tokensuncovering that CoT paths are frequently inherent in these sequences. Thisapproach not only bypasses the confounders of prompting but also allows us toassess the LLMs textitintrinsic reasoning abilities. Moreover we observethat the presence of a CoT in the decoding path correlates with a higherconfidence in the models decoded answer. This confidence metric effectivelydifferentiates between CoT and non-CoT paths. Extensive empirical studies onvarious reasoning benchmarks show that the proposed CoT-decoding substantiallyoutperforms the standard greedy decoding. |


| Item |Content|
| --- |---|
|idx| 2402.10196v1 |
|title| A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents |
|authors| Lingbo MoZeyi LiaoBoyuan ZhengYu SuChaowei XiaoHuan Sun
|links| http://arxiv.org/abs/2402.10196v1 |
|updated| 2024-02-15 18:51:32 UTC |
|summary| Language agents powered by large language models LLMs have seen explodingdevelopment. Their capability of using language as a vehicle for thought andcommunication lends an incredible level of flexibility and versatility. Peoplehave quickly capitalized on this capability to connect LLMs to a wide range ofexternal components and environments: databases tools the Internet roboticembodiment etc. Many believe an unprecedentedly powerful automation technologyis emerging. However new automation technologies come with new safety risksespecially for intricate systems like language agents. There is a surprisinglylarge gap between the speed and scale of their development and deployment andour understanding of their safety risks. Are we building a house of cards Inthis position paper we present the first systematic effort in mappingadversarial attacks against language agents. We first present a unifiedconceptual framework for agents with three major components: Perception Brainand Action. Under this framework we present a comprehensive discussion andpropose 12 potential attack scenarios against different components of an agentcovering different attack strategies e.g. input manipulation adversarialdemonstrations jailbreaking backdoors. We also draw connections tosuccessful attack strategies previously applied to LLMs. We emphasize theurgency to gain a thorough understanding of language agent risks before theirwidespread deployment. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.10210v1 |
|title| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation |
|authors| Huizhuo YuanZixiang ChenKaixuan JiQuanquan Gu
|links| http://arxiv.org/abs/2402.10210v1 |
|updated| 2024-02-15 18:59:18 UTC |
|summary| Fine-tuning Diffusion Models remains an underexplored frontier in generativeartificial intelligence GenAI especially when compared with the remarkableprogress made in fine-tuning Large Language Models LLMs. While cutting-edgediffusion models such as Stable Diffusion SD and SDXL rely on supervisedfine-tuning their performance inevitably plateaus after seeing a certainvolume of data. Recently reinforcement learning RL has been employed tofine-tune diffusion models with human preference data but it requires at leasttwo images winner and loser images for each text prompt. In this paperwe introduce an innovative technique called self-play fine-tuning for diffusionmodels SPIN-Diffusion where the diffusion model engages in competition withits earlier versions facilitating an iterative self-improvement process. Ourapproach offers an alternative to conventional supervised fine-tuning and RLstrategies significantly improving both model performance and alignment. Ourexperiments on the Pick-a-Pic dataset reveal that SPIN-Diffusion outperformsthe existing supervised fine-tuning method in aspects of human preferencealignment and visual appeal right from its first iteration. By the seconditeration it exceeds the performance of RLHF-based methods across all metricsachieving these results with less data. |


| Item |Content|
| --- |---|
|idx| 2402.10207v1 |
|title| Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment |
|authors| Rui YangXiaoman PanFeng LuoShuang QiuHan ZhongDong YuJianshu Chen
|links| http://arxiv.org/abs/2402.10207v1 |
|updated| 2024-02-15 18:58:31 UTC |
|summary| We consider the problem of multi-objective alignment of foundation modelswith human preferences which is a critical step towards helpful and harmlessAI systems. However it is generally costly and unstable to fine-tune largefoundation models using reinforcement learning RL and themulti-dimensionality heterogeneity and conflicting nature of humanpreferences further complicate the alignment process. In this paper weintroduce Rewards-in-Context RiC which conditions the response of afoundation model on multiple rewards in its prompt context and appliessupervised fine-tuning for alignment. The salient features of RiC aresimplicity and adaptivity as it only requires supervised fine-tuning of asingle foundation model and supports dynamic adjustment for user preferencesduring inference time. Inspired by the analytical solution of an abstractedconvex optimization problem our dynamic inference-time adjustment methodapproaches the Pareto-optimal solution for multiple objectives. Empiricalevidence demonstrates the efficacy of our method in aligning both LargeLanguage Models LLMs and diffusion models to accommodate diverse rewards withonly around 10 GPU hours compared with multi-objective RL baseline. |


| Item |Content|
| --- |---|
|idx| 2402.10206v1 |
|title| Ising on the Graph: Task-specific Graph Subsampling via the Ising Model |
|authors| Maria BånkestadJennifer AnderssonSebastian MairJens Sjölund
|links| http://arxiv.org/abs/2402.10206v1 |
|updated| 2024-02-15 18:58:18 UTC |
|summary| Reducing a graph while preserving its overall structure is an importantproblem with many applications. Typically the reduction approaches eitherremove edges sparsification or merge nodes coarsening in an unsupervisedway with no specific downstream task in mind. In this paper we present anapproach for subsampling graph structures using an Ising model defined oneither the nodes or edges and learning the external magnetic field of the Isingmodel using a graph neural network. Our approach is task-specific as it canlearn how to reduce a graph for a specific downstream task in an end-to-endfashion. The utilized loss function of the task does not even have to bedifferentiable. We showcase the versatility of our approach on three distinctapplications: image segmentation 3D shape sparsification and sparseapproximate matrix inverse determination. |


| Item |Content|
| --- |---|
|idx| 2402.10204v1 |
|title| Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model |
|authors| Mariia DrozdovaVitaliy KinakhOmkar BaitOlga TaranErica LastufkaMiroslava Dessauges-ZavadskyTaras HolotyakDaniel SchaererSlava Voloshynovskiy
|links| http://dx.doi.org/10.1051/0004-6361/202347948 |
|updated| 2024-02-15 18:57:24 UTC |
|summary| Reconstructing sky models from dirty radio images for accurate sourcelocalization and flux estimation is crucial for studying galaxy evolution athigh redshift especially in deep fields using instruments like the AtacamaLarge Millimetre Array ALMA. With new projects like the Square KilometreArray SKA theres a growing need for better source extraction methods.Current techniques such as CLEAN and PyBDSF often fail to detect faintsources highlighting the need for more accurate methods. This study proposesusing stochastic neural networks to rebuild sky models directly from dirtyimages. This method can pinpoint radio sources and measure their fluxes withrelated uncertainties marking a potential improvement in radio sourcecharacterization. We tested this approach on 10164 images simulated with theCASA tool simalma based on ALMAs Cycle 5.3 antenna setup. We appliedconditional Denoising Diffusion Probabilistic Models DDPMs for sky modelsreconstruction then used Photutils to determine source coordinates and fluxesassessing the models performance across different water vapor levels. Ourmethod showed excellence in source localization achieving more than 90completeness at a signal-to-noise ratio SNR as low as 2. It also surpassedPyBDSF in flux estimation accurately identifying fluxes for 96 of sources inthe test set a significant improvement over CLEAN PyBDSFs 57. ConditionalDDPMs is a powerful tool for image-to-image translation yielding accurate androbust characterisation of radio sources and outperforming existingmethodologies. While this study underscores its significant potential forapplications in radio astronomy we also acknowledge certain limitations thataccompany its usage suggesting directions for further refinement and research. |


| Item |Content|
| --- |---|
|idx| 2402.10196v1 |
|title| A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents |
|authors| Lingbo MoZeyi LiaoBoyuan ZhengYu SuChaowei XiaoHuan Sun
|links| http://arxiv.org/abs/2402.10196v1 |
|updated| 2024-02-15 18:51:32 UTC |
|summary| Language agents powered by large language models LLMs have seen explodingdevelopment. Their capability of using language as a vehicle for thought andcommunication lends an incredible level of flexibility and versatility. Peoplehave quickly capitalized on this capability to connect LLMs to a wide range ofexternal components and environments: databases tools the Internet roboticembodiment etc. Many believe an unprecedentedly powerful automation technologyis emerging. However new automation technologies come with new safety risksespecially for intricate systems like language agents. There is a surprisinglylarge gap between the speed and scale of their development and deployment andour understanding of their safety risks. Are we building a house of cards Inthis position paper we present the first systematic effort in mappingadversarial attacks against language agents. We first present a unifiedconceptual framework for agents with three major components: Perception Brainand Action. Under this framework we present a comprehensive discussion andpropose 12 potential attack scenarios against different components of an agentcovering different attack strategies e.g. input manipulation adversarialdemonstrations jailbreaking backdoors. We also draw connections tosuccessful attack strategies previously applied to LLMs. We emphasize theurgency to gain a thorough understanding of language agent risks before theirwidespread deployment. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.10211v1 |
|title| Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling |
|authors| Raunaq BhirangiChenyu WangVenkatesh PattabiramanCarmel MajidiAbhinav GuptaTess HellebrekersLerrel Pinto
|links| http://arxiv.org/abs/2402.10211v1 |
|updated| 2024-02-15 18:59:43 UTC |
|summary| Reasoning from sequences of raw sensory data is a ubiquitous problem acrossfields ranging from medical devices to robotics. These problems often involveusing long sequences of raw sensor data e.g. magnetometers piezoresistors topredict sequences of desirable physical quantities e.g. force inertialmeasurements. While classical approaches are powerful for locally-linearprediction problems they often fall short when using real-world sensors. Thesesensors are typically non-linear are affected by extraneous variables e.g.vibration and exhibit data-dependent drift. For many problems the predictiontask is exacerbated by small labeled datasets since obtaining ground-truthlabels requires expensive equipment. In this work we present HierarchicalState-Space Models HiSS a conceptually simple new technique for continuoussequential prediction. HiSS stacks structured state-space models on top of eachother to create a temporal hierarchy. Across six real-world sensor datasetsfrom tactile-based state prediction to accelerometer-based inertialmeasurement HiSS outperforms state-of-the-art sequence models such as causalTransformers LSTMs S4 and Mamba by at least 23 on MSE. Our experimentsfurther indicate that HiSS demonstrates efficient scaling to smaller datasetsand is compatible with existing data-filtering techniques. Code datasets andvideos can be found on https://hiss-csp.github.io. |


| Item |Content|
| --- |---|
|idx| 2402.10210v1 |
|title| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation |
|authors| Huizhuo YuanZixiang ChenKaixuan JiQuanquan Gu
|links| http://arxiv.org/abs/2402.10210v1 |
|updated| 2024-02-15 18:59:18 UTC |
|summary| Fine-tuning Diffusion Models remains an underexplored frontier in generativeartificial intelligence GenAI especially when compared with the remarkableprogress made in fine-tuning Large Language Models LLMs. While cutting-edgediffusion models such as Stable Diffusion SD and SDXL rely on supervisedfine-tuning their performance inevitably plateaus after seeing a certainvolume of data. Recently reinforcement learning RL has been employed tofine-tune diffusion models with human preference data but it requires at leasttwo images winner and loser images for each text prompt. In this paperwe introduce an innovative technique called self-play fine-tuning for diffusionmodels SPIN-Diffusion where the diffusion model engages in competition withits earlier versions facilitating an iterative self-improvement process. Ourapproach offers an alternative to conventional supervised fine-tuning and RLstrategies significantly improving both model performance and alignment. Ourexperiments on the Pick-a-Pic dataset reveal that SPIN-Diffusion outperformsthe existing supervised fine-tuning method in aspects of human preferencealignment and visual appeal right from its first iteration. By the seconditeration it exceeds the performance of RLHF-based methods across all metricsachieving these results with less data. |


| Item |Content|
| --- |---|
|idx| 2402.10208v1 |
|title| Recovering the Pre-Fine-Tuning Weights of Generative Models |
|authors| Eliahu HorwitzJonathan KahanaYedid Hoshen
|links| http://arxiv.org/abs/2402.10208v1 |
|updated| 2024-02-15 18:59:02 UTC |
|summary| The dominant paradigm in generative modeling consists of two steps: ipre-training on a large-scale but unsafe dataset ii aligning the pre-trainedmodel with human values via fine-tuning. This practice is considered safe asno current method can recover the unsafe pre-fine-tuning model weights. Inthis paper we demonstrate that this assumption is often false. Concretely wepresent Spectral DeTuning a method that can recover the weights of thepre-fine-tuning model using a few low-rank LoRA fine-tuned models. Incontrast to previous attacks that attempt to recover pre-fine-tuningcapabilities our method aims to recover the exact pre-fine-tuning weights. Ourapproach exploits this new vulnerability against large-scale models such as apersonalized Stable Diffusion and an aligned Mistral. |


| Item |Content|
| --- |---|
|idx| 2402.10207v1 |
|title| Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment |
|authors| Rui YangXiaoman PanFeng LuoShuang QiuHan ZhongDong YuJianshu Chen
|links| http://arxiv.org/abs/2402.10207v1 |
|updated| 2024-02-15 18:58:31 UTC |
|summary| We consider the problem of multi-objective alignment of foundation modelswith human preferences which is a critical step towards helpful and harmlessAI systems. However it is generally costly and unstable to fine-tune largefoundation models using reinforcement learning RL and themulti-dimensionality heterogeneity and conflicting nature of humanpreferences further complicate the alignment process. In this paper weintroduce Rewards-in-Context RiC which conditions the response of afoundation model on multiple rewards in its prompt context and appliessupervised fine-tuning for alignment. The salient features of RiC aresimplicity and adaptivity as it only requires supervised fine-tuning of asingle foundation model and supports dynamic adjustment for user preferencesduring inference time. Inspired by the analytical solution of an abstractedconvex optimization problem our dynamic inference-time adjustment methodapproaches the Pareto-optimal solution for multiple objectives. Empiricalevidence demonstrates the efficacy of our method in aligning both LargeLanguage Models LLMs and diffusion models to accommodate diverse rewards withonly around 10 GPU hours compared with multi-objective RL baseline. |


| Item |Content|
| --- |---|
|idx| 2402.10206v1 |
|title| Ising on the Graph: Task-specific Graph Subsampling via the Ising Model |
|authors| Maria BånkestadJennifer AnderssonSebastian MairJens Sjölund
|links| http://arxiv.org/abs/2402.10206v1 |
|updated| 2024-02-15 18:58:18 UTC |
|summary| Reducing a graph while preserving its overall structure is an importantproblem with many applications. Typically the reduction approaches eitherremove edges sparsification or merge nodes coarsening in an unsupervisedway with no specific downstream task in mind. In this paper we present anapproach for subsampling graph structures using an Ising model defined oneither the nodes or edges and learning the external magnetic field of the Isingmodel using a graph neural network. Our approach is task-specific as it canlearn how to reduce a graph for a specific downstream task in an end-to-endfashion. The utilized loss function of the task does not even have to bedifferentiable. We showcase the versatility of our approach on three distinctapplications: image segmentation 3D shape sparsification and sparseapproximate matrix inverse determination. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.10210v1 |
|title| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation |
|authors| Huizhuo YuanZixiang ChenKaixuan JiQuanquan Gu
|links| http://arxiv.org/abs/2402.10210v1 |
|updated| 2024-02-15 18:59:18 UTC |
|summary| Fine-tuning Diffusion Models remains an underexplored frontier in generativeartificial intelligence GenAI especially when compared with the remarkableprogress made in fine-tuning Large Language Models LLMs. While cutting-edgediffusion models such as Stable Diffusion SD and SDXL rely on supervisedfine-tuning their performance inevitably plateaus after seeing a certainvolume of data. Recently reinforcement learning RL has been employed tofine-tune diffusion models with human preference data but it requires at leasttwo images winner and loser images for each text prompt. In this paperwe introduce an innovative technique called self-play fine-tuning for diffusionmodels SPIN-Diffusion where the diffusion model engages in competition withits earlier versions facilitating an iterative self-improvement process. Ourapproach offers an alternative to conventional supervised fine-tuning and RLstrategies significantly improving both model performance and alignment. Ourexperiments on the Pick-a-Pic dataset reveal that SPIN-Diffusion outperformsthe existing supervised fine-tuning method in aspects of human preferencealignment and visual appeal right from its first iteration. By the seconditeration it exceeds the performance of RLHF-based methods across all metricsachieving these results with less data. |


| Item |Content|
| --- |---|
|idx| 2402.10208v1 |
|title| Recovering the Pre-Fine-Tuning Weights of Generative Models |
|authors| Eliahu HorwitzJonathan KahanaYedid Hoshen
|links| http://arxiv.org/abs/2402.10208v1 |
|updated| 2024-02-15 18:59:02 UTC |
|summary| The dominant paradigm in generative modeling consists of two steps: ipre-training on a large-scale but unsafe dataset ii aligning the pre-trainedmodel with human values via fine-tuning. This practice is considered safe asno current method can recover the unsafe pre-fine-tuning model weights. Inthis paper we demonstrate that this assumption is often false. Concretely wepresent Spectral DeTuning a method that can recover the weights of thepre-fine-tuning model using a few low-rank LoRA fine-tuned models. Incontrast to previous attacks that attempt to recover pre-fine-tuningcapabilities our method aims to recover the exact pre-fine-tuning weights. Ourapproach exploits this new vulnerability against large-scale models such as apersonalized Stable Diffusion and an aligned Mistral. |


| Item |Content|
| --- |---|
|idx| 2402.10204v1 |
|title| Radio-astronomical Image Reconstruction with Conditional Denoising Diffusion Model |
|authors| Mariia DrozdovaVitaliy KinakhOmkar BaitOlga TaranErica LastufkaMiroslava Dessauges-ZavadskyTaras HolotyakDaniel SchaererSlava Voloshynovskiy
|links| http://dx.doi.org/10.1051/0004-6361/202347948 |
|updated| 2024-02-15 18:57:24 UTC |
|summary| Reconstructing sky models from dirty radio images for accurate sourcelocalization and flux estimation is crucial for studying galaxy evolution athigh redshift especially in deep fields using instruments like the AtacamaLarge Millimetre Array ALMA. With new projects like the Square KilometreArray SKA theres a growing need for better source extraction methods.Current techniques such as CLEAN and PyBDSF often fail to detect faintsources highlighting the need for more accurate methods. This study proposesusing stochastic neural networks to rebuild sky models directly from dirtyimages. This method can pinpoint radio sources and measure their fluxes withrelated uncertainties marking a potential improvement in radio sourcecharacterization. We tested this approach on 10164 images simulated with theCASA tool simalma based on ALMAs Cycle 5.3 antenna setup. We appliedconditional Denoising Diffusion Probabilistic Models DDPMs for sky modelsreconstruction then used Photutils to determine source coordinates and fluxesassessing the models performance across different water vapor levels. Ourmethod showed excellence in source localization achieving more than 90completeness at a signal-to-noise ratio SNR as low as 2. It also surpassedPyBDSF in flux estimation accurately identifying fluxes for 96 of sources inthe test set a significant improvement over CLEAN PyBDSFs 57. ConditionalDDPMs is a powerful tool for image-to-image translation yielding accurate androbust characterisation of radio sources and outperforming existingmethodologies. While this study underscores its significant potential forapplications in radio astronomy we also acknowledge certain limitations thataccompany its usage suggesting directions for further refinement and research. |


| Item |Content|
| --- |---|
|idx| 2402.10130v1 |
|title| Is Continual Learning Ready for Real-world Challenges? |
|authors| Theodora KontogianniYuanwen YueSiyu TangKonrad Schindler
|links| http://arxiv.org/abs/2402.10130v1 |
|updated| 2024-02-15 17:34:56 UTC |
|summary| Despite continual learnings long and well-established academic history itsapplication in real-world scenarios remains rather limited. This paper contendsthat this gap is attributable to a misalignment between the actual challengesof continual learning and the evaluation protocols in use rendering proposedsolutions ineffective for addressing the complexities of real-world setups. Wevalidate our hypothesis and assess progress to date using a new 3D semanticsegmentation benchmark OCL-3DSS. We investigate various continual learningschemes from the literature by utilizing more realistic protocols thatnecessitate online and continual learning for dynamic real-world scenarioseg. in robotics and 3D vision applications. The outcomes are sobering: allconsidered methods perform poorly significantly deviating from the upper boundof joint offline training. This raises questions about the applicability ofexisting methods in realistic settings. Our paper aims to initiate a paradigmshift advocating for the adoption of continual learning methods through newexperimental protocols that better emulate real-world conditions to facilitatebreakthroughs in the field. |


| Item |Content|
| --- |---|
|idx| 2402.10128v1 |
|title| GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering |
|authors| Abdullah HamdiLuke Melas-KyriaziGuocheng QianJinjie MaiRuoshi LiuCarl VondrickBernard GhanemAndrea Vedaldi
|links| http://arxiv.org/abs/2402.10128v1 |
|updated| 2024-02-15 17:32:50 UTC |
|summary| Advancements in 3D Gaussian Splatting have significantly accelerated 3Dreconstruction and generation. However it may require a large number ofGaussians which creates a substantial memory footprint. This paper introducesGES Generalized Exponential Splatting a novel representation that employsGeneralized Exponential Function GEF to model 3D scenes requiring far fewerparticles to represent a scene and thus significantly outperforming GaussianSplatting methods in efficiency with a plug-and-play replacement ability forGaussian-based utilities. GES is validated theoretically and empirically inboth principled 1D setup and realistic 3D scenes.  It is shown to represent signals with sharp edges more accurately which aretypically challenging for Gaussians due to their inherent low-passcharacteristics. Our empirical analysis demonstrates that GEF outperformsGaussians in fitting natural-occurring signals e.g. squares triangles andparabolic signals thereby reducing the need for extensive splittingoperations that increase the memory footprint of Gaussian Splatting. With theaid of a frequency-modulated loss GES achieves competitive performance innovel-view synthesis benchmarks while requiring less than half the memorystorage of Gaussian Splatting and increasing the rendering speed by up to 39.The code is available on the project website https://abdullahamdi.com/ges . |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.10210v1 |
|title| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation |
|authors| Huizhuo YuanZixiang ChenKaixuan JiQuanquan Gu
|links| http://arxiv.org/abs/2402.10210v1 |
|updated| 2024-02-15 18:59:18 UTC |
|summary| Fine-tuning Diffusion Models remains an underexplored frontier in generativeartificial intelligence GenAI especially when compared with the remarkableprogress made in fine-tuning Large Language Models LLMs. While cutting-edgediffusion models such as Stable Diffusion SD and SDXL rely on supervisedfine-tuning their performance inevitably plateaus after seeing a certainvolume of data. Recently reinforcement learning RL has been employed tofine-tune diffusion models with human preference data but it requires at leasttwo images winner and loser images for each text prompt. In this paperwe introduce an innovative technique called self-play fine-tuning for diffusionmodels SPIN-Diffusion where the diffusion model engages in competition withits earlier versions facilitating an iterative self-improvement process. Ourapproach offers an alternative to conventional supervised fine-tuning and RLstrategies significantly improving both model performance and alignment. Ourexperiments on the Pick-a-Pic dataset reveal that SPIN-Diffusion outperformsthe existing supervised fine-tuning method in aspects of human preferencealignment and visual appeal right from its first iteration. By the seconditeration it exceeds the performance of RLHF-based methods across all metricsachieving these results with less data. |


| Item |Content|
| --- |---|
|idx| 2402.10198v1 |
|title| Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention |
|authors| Romain IlbertAmbroise OdonnatVasilii FeofanovAladin VirmauxGiuseppe PaoloThemis PalpanasIevgen Redko
|links| http://arxiv.org/abs/2402.10198v1 |
|updated| 2024-02-15 18:55:05 UTC |
|summary| Transformer-based architectures achieved breakthrough performance in naturallanguage processing and computer vision yet they remain inferior to simplerlinear baselines in multivariate long-term forecasting. To better understandthis phenomenon we start by studying a toy linear forecasting problem forwhich we show that transformers are incapable of converging to their truesolution despite their high expressive power. We further identify the attentionof transformers as being responsible for this low generalization capacity.Building upon this insight we propose a shallow lightweight transformer modelthat successfully escapes bad local minima when optimized with sharpness-awareoptimization. We empirically demonstrate that this result extends to allcommonly used real-world multivariate time series datasets. In particularSAMformer surpasses the current state-of-the-art model TSMixer by 14.33 onaverage while having 4 times fewer parameters. The code is available athttps://github.com/romilbert/samformer. |


| Item |Content|
| --- |---|
|idx| 2402.10127v1 |
|title| Nonlinear spiked covariance matrices and signal propagation in deep neural networks |
|authors| Zhichao WangDenny WuZhou Fan
|links| http://arxiv.org/abs/2402.10127v1 |
|updated| 2024-02-15 17:31:19 UTC |
|summary| Many recent works have studied the eigenvalue spectrum of the ConjugateKernel CK defined by the nonlinear feature map of a feedforward neuralnetwork. However existing results only establish weak convergence of theempirical eigenvalue distribution and fall short of providing precisequantitative characterizations of the spike eigenvalues and eigenvectorsthat often capture the low-dimensional signal structure of the learningproblem. In this work we characterize these signal eigenvalues andeigenvectors for a nonlinear version of the spiked covariance model includingthe CK as a special case. Using this general result we give a quantitativedescription of how spiked eigenstructure in the input data propagates throughthe hidden layers of a neural network with random weights. As a secondapplication we study a simple regime of representation learning where theweight matrix develops a rank-one signal component over training andcharacterize the alignment of the target function with the spike eigenvector ofthe CK on test data. |


| Item |Content|
| --- |---|
|idx| 2402.10065v1 |
|title| How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage |
|authors| Achraf AzizeDebabrota Basu
|links| http://arxiv.org/abs/2402.10065v1 |
|updated| 2024-02-15 16:30:55 UTC |
|summary| We study the per-datum Membership Inference Attacks MIAs where an attackeraims to infer whether a fixed target datum has been included in the inputdataset of an algorithm and thus violates privacy. First we define themembership leakage of a datum as the advantage of the optimal adversarytargeting to identify it. Then we quantify the per-datum membership leakagefor the empirical mean and show that it depends on the Mahalanobis distancebetween the target datum and the data-generating distribution. We furtherassess the effect of two privacy defences i.e. adding Gaussian noise andsub-sampling. We quantify exactly how both of them decrease the per-datummembership leakage. Our analysis builds on a novel proof technique thatcombines an Edgeworth expansion of the likelihood ratio test and aLindeberg-Feller central limit theorem. Our analysis connects the existinglikelihood ratio and scalar product attacks and also justifies differentcanary selection strategies used in the privacy auditing literature. Finallyour experiments demonstrate the impacts of the leakage score the sub-samplingratio and the noise scale on the per-datum membership leakage as indicated bythe theory. |


| Item |Content|
| --- |---|
|idx| 2402.10043v1 |
|title| How to validate average calibration for machine learning regression tasks ? |
|authors| Pascal Pernot
|links| http://arxiv.org/abs/2402.10043v1 |
|updated| 2024-02-15 16:05:35 UTC |
|summary| Average calibration of the uncertainties of machine learning regression taskscan be tested in two ways. One way is to estimate the calibration error CE asthe difference between the mean absolute error MSE and the mean variance MVor mean squared uncertainty. The alternative is to compare the mean squaredz-scores or scaled errors ZMS to 1. Both approaches might lead to differentconclusion as illustrated on an ensemble of datasets from the recent machinelearning uncertainty quantification literature. It is shown here that the CE isvery sensitive to the distribution of uncertainties and notably to thepresence of outlying uncertainties and that it cannot be used reliably forcalibration testing. By contrast the ZMS statistic does not present thissensitivity issue and offers the most reliable approach in this context.Implications for the validation of conditional calibration are discussed. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.10050v1 |
|title| On-Demand Myoelectric Control Using Wake Gestures to Eliminate False Activations During Activities of Daily Living |
|authors| Ethan EddyEvan CampbellScott BatemanErik Scheme
|links| http://arxiv.org/abs/2402.10050v1 |
|updated| 2024-02-15 16:11:47 UTC |
|summary| While myoelectric control has recently become a focus of increased researchas a possible flexible hands-free input modality current control approachesare prone to inadvertent false activations in real-world conditions. In thiswork a novel myoelectric control paradigm -- on-demand myoelectric control --is proposed designed and evaluated to reduce the number of unrelated musclemovements that are incorrectly interpreted as input gestures . By leveragingthe concept of wake gestures users were able to switch between a dedicatedcontrol mode and a sleep mode effectively eliminating inadvertent activationsduring activities of daily living ADLs. The feasibility of wake gestures wasdemonstrated in this work through two online ubiquitous EMG control tasks withvarying difficulty levels dismissing an alarm and controlling a robot. Theproposed control scheme was able to appropriately ignore almost allnon-targeted muscular inputs during ADLs 99.9 while maintaining sufficientsensitivity for reliable mode switching during intentional wake gestureelicitation. These results highlight the potential of wake gestures as acritical step towards enabling ubiquitous myoelectric control-based on-demandinput for a wide range of applications. |


| Item |Content|
| --- |---|
|idx| 2402.09990v1 |
|title| TIAViz: A Browser-based Visualization Tool for Computational Pathology Models |
|authors| Mark EastwoodJohn PocockMostafa JahanifarAdam ShephardSkiros HabibEthar AlzaidAbdullah AlsalemiJan Lukas RobertusNasir RajpootShan RazaFayyaz Minhas
|links| http://arxiv.org/abs/2402.09990v1 |
|updated| 2024-02-15 14:54:46 UTC |
|summary| Digital pathology has gained significant traction in modern healthcaresystems. This shift from optical microscopes to digital imagery brings with itthe potential for improved diagnosis efficiency and the integration of AItools into the pathologists workflow. A critical aspect of this isvisualization. Throughout the development of a machine learning ML model indigital pathology it is crucial to have flexible openly available tools tovisualize models from their outputs and predictions to the underlyingannotations and images used to train or test a model. We introduce TIAViz aPython-based visualization tool built into TIAToolbox which allows flexibleinteractive fully zoomable overlay of a wide variety of information onto wholeslide images including graphs heatmaps segmentations annotations and otherWSIs. The UI is browser-based allowing use either locally on a remotemachine or on a server to provide publicly available demos. This tool is opensource and is made available at:https://github.com/TissueImageAnalytics/tiatoolbox and via pip installationpip install tiatoolbox and conda as part of TIAToolbox. |


| Item |Content|
| --- |---|
|idx| 2402.09939v1 |
|title| Generative AI in the Construction Industry: A State-of-the-art Analysis |
|authors| Ridwan TaiwoIdris Temitope BelloSulemana Fatoama AbdulaiAbdul-Mugis YussifBabatunde Abiodun SalamiAbdullahi SakaTarek Zayed
|links| http://arxiv.org/abs/2402.09939v1 |
|updated| 2024-02-15 13:39:55 UTC |
|summary| The construction industry is a vital sector of the global economy but itfaces many productivity challenges in various processes such as designplanning procurement inspection and maintenance. Generative artificialintelligence AI which can create novel and realistic data or content suchas text image video or code based on some input or prior knowledge offersinnovative and disruptive solutions to address these challenges. However thereis a gap in the literature on the current state opportunities and challengesof generative AI in the construction industry. This study aims to fill this gapby providing a state-of-the-art analysis of generative AI in construction withthree objectives: 1 to review and categorize the existing and emerginggenerative AI opportunities and challenges in the construction industry 2 topropose a framework for construction firms to build customized generative AIsolutions using their own data comprising steps such as data collectiondataset curation training custom large language model LLM model evaluationand deployment and 3 to demonstrate the framework via a case study ofdeveloping a generative model for querying contract documents. The results showthat retrieval augmented generation RAG improves the baseline LLM by 5.29.4 and 4.8 in terms of quality relevance and reproducibility. This studyprovides academics and construction professionals with a comprehensive analysisand practical framework to guide the adoption of generative AI techniques toenhance productivity quality safety and sustainability across theconstruction industry. |


| Item |Content|
| --- |---|
|idx| 2402.09894v1 |
|title| Not Just Novelty: A Longitudinal Study on Utility and Customization of AI Workflows |
|authors| Tao LongKaty Ilonka GeroLydia B. Chilton
|links| http://arxiv.org/abs/2402.09894v1 |
|updated| 2024-02-15 11:39:11 UTC |
|summary| Generative AI brings novel and impressive abilities to help people ineveryday tasks. There are many AI workflows that solve real and complexproblems by chaining AI outputs together with human interaction. Although thereis an undeniable lure of AI its uncertain how useful generative AI workflowsare after the novelty wears off. Additionally tools built with generative AIhave the potential to be personalized and adapted quickly and easily but dousers take advantage of the potential to customize We conducted a three-weeklongitudinal study with 12 users to understand the familiarization andcustomization of generative AI tools for science communication. Our studyrevealed that the familiarization phase lasts for 4.3 sessions where usersexplore the capabilities of the workflow and which aspects they find useful.After familiarization the perceived utility of the system is rated higher thanbefore indicating that the perceived utility of AI is not just a noveltyeffect. The increase in benefits mainly comes from end-users ability tocustomize prompts and thus appropriate the system to their own needs. Thispoints to a future where generative AI systems can allow us to design forappropriation. |


| Item |Content|
| --- |---|
|idx| 2402.09880v1 |
|title| Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence |
|authors| Timothy R. McIntoshTeo SusnjakTong LiuPaul WattersMalka N. Halgamuge
|links| http://arxiv.org/abs/2402.09880v1 |
|updated| 2024-02-15 11:08:10 UTC |
|summary| The rapid rise in popularity of Large Language Models LLMs with emergingcapabilities has spurred public curiosity to evaluate and compare differentLLMs leading many researchers to propose their LLM benchmarks. Noticingpreliminary inadequacies in those benchmarks we embarked on a study tocritically assess 23 state-of-the-art LLM benchmarks using our novel unifiedevaluation framework through the lenses of people process and technologyunder the pillars of functionality and security. Our research uncoveredsignificant limitations including biases difficulties in measuring genuinereasoning adaptability implementation inconsistencies prompt engineeringcomplexity evaluator diversity and the overlooking of cultural andideological norms in one comprehensive assessment. Our discussions emphasizedthe urgent need for standardized methodologies regulatory certainties andethical guidelines in light of Artificial Intelligence AI advancementsincluding advocating for an evolution from static benchmarks to dynamicbehavioral profiling to accurately capture LLMs complex behaviors andpotential risks. Our study highlighted the necessity for a paradigm shift inLLM evaluation methodologies underlining the importance of collaborativeefforts for the development of universally accepted benchmarks and theenhancement of AI systems integration into society. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.10172v1 |
|title| OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models |
|authors| Ali AhmadiTeshniziWenzhi GaoMadeleine Udell
|links| http://arxiv.org/abs/2402.10172v1 |
|updated| 2024-02-15 18:19:18 UTC |
|summary| Optimization problems are pervasive in sectors from manufacturing anddistribution to healthcare. However most such problems are still solvedheuristically by hand rather than optimally by state-of-the-art solvers becausethe expertise required to formulate and solve these problems limits thewidespread adoption of optimization tools and techniques. This paper introducesOptiMUS a Large Language Model LLM-based agent designed to formulate andsolve mixed integer linear programming problems from their natural languagedescriptions. OptiMUS can develop mathematical models write and debug solvercode evaluate the generated solutions and improve its model and code based onthese evaluations. OptiMUS utilizes a modular structure to process problemsallowing it to handle problems with long descriptions and complex data withoutlong prompts. Experiments demonstrate that OptiMUS outperforms existingstate-of-the-art methods on easy datasets by more than 20 and on harddatasets including a new dataset NLP4LP released with this paper thatfeatures long and complex problems by more than 30. |


| Item |Content|
| --- |---|
|idx| 2402.09921v1 |
|title| Identifying and modelling cognitive biases in mobility choices |
|authors| Chloe ConradCarole Adam
|links| http://arxiv.org/abs/2402.09921v1 |
|updated| 2024-02-15 12:58:27 UTC |
|summary| This report presents results from an M1 internship dedicated to agent-basedmodelling and simulation of daily mobility choices. This simulation is intendedto be realistic enough to serve as a basis for a serious game about themobility transition. In order to ensure this level of realism we conducted asurvey to measure if real mobility choices are made rationally or how biasedthey are. Results analysed here show that various biases could play a role indecisions. We then propose an implementation in a GAMA agent-based simulation. |


| Item |Content|
| --- |---|
|idx| 2402.09776v1 |
|title| Strategic Vote Timing in Online Elections With Public Tallies |
|authors| Aviv YaishSvetlana AbramovaRainer Böhme
|links| http://arxiv.org/abs/2402.09776v1 |
|updated| 2024-02-15 08:06:21 UTC |
|summary| We study the effect of public tallies on online elections in a setting wherevoting is costly and voters are allowed to strategically time their votes. Thestrategic importance of choosing emphwhen to vote arises when votes arepublic such as in online event scheduling polls e.g. Doodle or inblockchain governance mechanisms. In particular there is a tension betweenvoting early to influence future votes and waiting to observe interim resultsand avoid voting costs if the outcome has already been decided.  Our study draws on empirical findings showing that temporal bandwagoneffects occur when interim results are revealed to the electorate: late votersare more likely to vote for leading candidates. To capture this phenomenon weanalyze a novel model where the electorate consists of informed voters who havea preferred candidate and uninformed swing voters who can be swayed accordingto the interim outcome at the time of voting. In our main results we prove theexistence of equilibria where both early and late voting occur with a positiveprobability and we characterize conditions that lead to the appearance oflast minute voting behavior where all informed voters vote late. |


| Item |Content|
| --- |---|
|idx| 2402.09714v1 |
|title| An Accelerated Distributed Stochastic Gradient Method with Momentum |
|authors| Kun HuangShi PuAngelia Nedić
|links| http://arxiv.org/abs/2402.09714v1 |
|updated| 2024-02-15 05:15:22 UTC |
|summary| In this paper we introduce an accelerated distributed stochastic gradientmethod with momentum for solving the distributed optimization problem where agroup of n agents collaboratively minimize the average of the local objectivefunctions over a connected network. The method termed Distributed StochasticMomentum Tracking DSMT is a single-loop algorithm that utilizes themomentum tracking technique as well as the Loopless Chebyshev AccelerationLCA method. We show that DSMT can asymptotically achieve comparableconvergence rates as centralized stochastic gradient descent SGD method undera general variance condition regarding the stochastic gradients. Moreover thenumber of iterations transient times required for DSMT to achieve such ratesbehaves as mathcalOn5/3/1-lambda for minimizing general smoothobjective functions and mathcalOsqrtn/1-lambda under thePolyak-Lojasiewicz PL condition. Here the term 1-lambda denotes thespectral gap of the mixing matrix related to the underlying network topology.Notably the obtained results do not rely on multiple inter-node communicationsor stochastic gradient accumulation per iteration and the transient times arethe shortest under the setting to the best of our knowledge. |


| Item |Content|
| --- |---|
|idx| 2402.09563v1 |
|title| ABIDES-Economist: Agent-Based Simulation of Economic Systems with Learning Agents |
|authors| Kshama DwarakanathSvitlana VyetrenkoPeyman TavallaliTucker Balch
|links| http://arxiv.org/abs/2402.09563v1 |
|updated| 2024-02-14 20:26:52 UTC |
|summary| We introduce a multi-agent simulator for economic systems comprised ofheterogeneous Households heterogeneous Firms Central Bank and Governmentagents that could be subjected to exogenous stochastic shocks. Theinteraction between agents defines the production and consumption of goods inthe economy alongside the flow of money. Each agent can be designed to actaccording to fixed rule-based strategies or learn their strategies usinginteractions with others in the simulator. We ground our simulator by choosingagent heterogeneity parameters based on economic literature while designingtheir action spaces in accordance with real data in the United States. Oursimulator facilitates the use of reinforcement learning strategies for theagents via an OpenAI Gym style environment definition for the economic system.We demonstrate the utility of our simulator by simulating and analyzing twohypothetical yet interesting economic scenarios. The first scenarioinvestigates the impact of heterogeneous household skills on their learnedpreferences to work at different firms. The second scenario examines the impactof a positive production shock to one of two firms on its pricing strategy incomparison to the second firm. We aspire that our platform sets a stage forsubsequent research at the intersection of artificial intelligence andeconomics. |


