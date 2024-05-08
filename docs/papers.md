# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.04532v1 |
|title| QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving |
|authors| Yujun LinHaotian TangShang YangZhekai ZhangGuangxuan XiaoChuang GanSong Han
|links| http://arxiv.org/abs/2405.04532v1 |
|updated| 2024-05-07 17:59:30 UTC |
|summary| Quantization can accelerate large language model LLM inference. Goingbeyond INT8 quantization the research community is actively exploring evenlower precision such as INT4. Nonetheless state-of-the-art INT4 quantizationtechniques only accelerate low-batch edge LLM inference failing to deliverperformance gains in large-batch cloud-based LLM serving. We uncover acritical issue: existing INT4 quantization methods suffer from significantruntime overhead 20-90 when dequantizing either weights or partial sums onGPUs. To address this challenge we introduce QoQ a W4A8KV4 quantizationalgorithm with 4-bit weight 8-bit activation and 4-bit KV cache. QoQ standsfor quattuor-octo-quattuor which represents 4-8-4 in Latin. QoQ is implementedby the QServe inference library that achieves measured speedup. The key insightdriving QServe is that the efficiency of LLM serving on GPUs is criticallyinfluenced by operations on low-throughput CUDA cores. Building upon thisinsight in QoQ algorithm we introduce progressive quantization that can allowlow dequantization overhead in W4A8 GEMM. Additionally we developSmoothAttention to effectively mitigate the accuracy degradation incurred by4-bit KV quantization. In the QServe system we perform compute-aware weightreordering and take advantage of register-level parallelism to reducedequantization latency. We also make fused attention memory-bound harnessingthe performance gain brought by KV4 quantization. As a result QServe improvesthe maximum achievable serving throughput of Llama-3-8B by 1.2x on A100 1.4xon L40S and Qwen1.5-72B by 2.4x on A100 3.5x on L40S compared toTensorRT-LLM. Remarkably QServe on L40S GPU can achieve even higher throughputthan TensorRT-LLM on A100. Thus QServe effectively reduces the dollar cost ofLLM serving by 3x. Code is available at https://github.com/mit-han-lab/qserve. |


| Item |Content|
| --- |---|
|idx| 2405.04520v1 |
|title| NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts |
|authors| Shudan ZhangHanlin ZhaoXiao LiuQinkai ZhengZehan QiXiaotao GuXiaohan ZhangYuxiao DongJie Tang
|links| http://arxiv.org/abs/2405.04520v1 |
|updated| 2024-05-07 17:52:51 UTC |
|summary| Large language models LLMs have manifested strong ability to generate codesfor productive activities. However current benchmarks for code synthesis suchas HumanEval MBPP and DS-1000 are predominantly oriented towardsintroductory tasks on algorithm and data science insufficiently satisfyingchallenging requirements prevalent in real-world coding. To fill this gap wepropose NaturalCodeBench NCB a challenging code benchmark designed to mirrorthe complexity and variety of scenarios in real coding tasks. NCB comprises 402high-quality problems in Python and Java meticulously selected from naturaluser queries from online coding services covering 6 different domains. Notingthe extraordinary difficulty in creating testing cases for real-world querieswe also introduce a semi-automated pipeline to enhance the efficiency of testcase construction. Comparing with manual solutions it achieves an efficiencyincrease of more than 4 times. Our systematic experiments on 39 LLMs find thatperformance gaps on NCB between models with close HumanEval scores could stillbe significant indicating a lack of focus on practical code synthesisscenarios or over-specified optimization on HumanEval. On the other hand eventhe best-performing GPT-4 is still far from satisfying on NCB. The evaluationtoolkit and development set are available athttps://github.com/THUDM/NaturalCodeBench. |


| Item |Content|
| --- |---|
|idx| 2405.04515v1 |
|title| A Transformer with Stack Attention |
|authors| Jiaoda LiJennifer C. WhiteMrinmaya SachanRyan Cotterell
|links| http://arxiv.org/abs/2405.04515v1 |
|updated| 2024-05-07 17:47:57 UTC |
|summary| Natural languages are believed to be mildly context-sensitive. Despiteunderpinning remarkably capable large language models transformers are unableto model many context-free language tasks. In an attempt to address thislimitation in the modeling power of transformer-based language models wepropose augmenting them with a differentiable stack-based attention mechanism.Our stack-based attention mechanism can be incorporated into anytransformer-based language model and adds a level of interpretability to themodel. We show that the addition of our stack-based attention mechanism enablesthe transformer to model some but not all deterministic context-freelanguages. |


| Item |Content|
| --- |---|
|idx| 2405.04513v1 |
|title| Switchable Decision: Dynamic Neural Generation Networks |
|authors| Shujian ZhangKorawat TanwisuthChengyue GongPengcheng HeMingyuan Zhou
|links| http://arxiv.org/abs/2405.04513v1 |
|updated| 2024-05-07 17:44:54 UTC |
|summary| Auto-regressive generation models achieve competitive performance across manydifferent NLP tasks such as summarization question answering andclassifications. However they are also known for being slow in inferencewhich makes them challenging to deploy in real-time applications. We propose aswitchable decision to accelerate inference by dynamically assigningcomputation resources for each data instance. Automatically making decisions onwhere to skip and how to balance quality and computation cost with constrainedoptimization our dynamic neural generation networks enforce the efficientinference path and determine the optimized trade-off. Experiments acrossquestion answering summarization and classification benchmarks show that ourmethod benefits from less computation cost during inference while keeping thesame accuracy. Extensive experiments and ablation studies demonstrate that ourmethod can be general effective and beneficial for many NLP tasks. |


| Item |Content|
| --- |---|
|idx| 2405.04495v1 |
|title| Toward In-Context Teaching: Adapting Examples to Students' Misconceptions |
|authors| Alexis RossJacob Andreas
|links| http://arxiv.org/abs/2405.04495v1 |
|updated| 2024-05-07 17:05:27 UTC |
|summary| When a teacher provides examples for a student to study these examples mustbe informative enabling a student to progress from their current state towarda target concept or skill. Good teachers must therefore simultaneously inferwhat students already know and adapt their teaching to students changing stateof knowledge. There is increasing interest in using computational modelsparticularly large language models as pedagogical tools. As students languagemodels in particular have shown a remarkable ability to adapt to new tasksgiven small numbers of examples. But how effectively can these models adapt asteachers to students of different types To study this question we introduce asuite of models and evaluation methods we call AdapT. AdapT has two components:1 a collection of simulated Bayesian student models that can be used forevaluation of automated teaching methods 2 a platform for evaluation withhuman students to characterize the real-world effectiveness of these methods.We additionally introduce 3 AToM a new probabilistic model for adaptiveteaching that jointly infers students past beliefs and optimizes for thecorrectness of future beliefs. In evaluations of simulated students acrossthree learning domains fraction arithmetic English morphology functionlearning AToM systematically outperforms LLM-based and standard Bayesianteaching models. In human experiments both AToM and LLMs outperformnon-adaptive random example selection. Our results highlight both thedifficulty of the adaptive teaching task and the potential of learned adaptivemodels for solving it. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.04532v1 |
|title| QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving |
|authors| Yujun LinHaotian TangShang YangZhekai ZhangGuangxuan XiaoChuang GanSong Han
|links| http://arxiv.org/abs/2405.04532v1 |
|updated| 2024-05-07 17:59:30 UTC |
|summary| Quantization can accelerate large language model LLM inference. Goingbeyond INT8 quantization the research community is actively exploring evenlower precision such as INT4. Nonetheless state-of-the-art INT4 quantizationtechniques only accelerate low-batch edge LLM inference failing to deliverperformance gains in large-batch cloud-based LLM serving. We uncover acritical issue: existing INT4 quantization methods suffer from significantruntime overhead 20-90 when dequantizing either weights or partial sums onGPUs. To address this challenge we introduce QoQ a W4A8KV4 quantizationalgorithm with 4-bit weight 8-bit activation and 4-bit KV cache. QoQ standsfor quattuor-octo-quattuor which represents 4-8-4 in Latin. QoQ is implementedby the QServe inference library that achieves measured speedup. The key insightdriving QServe is that the efficiency of LLM serving on GPUs is criticallyinfluenced by operations on low-throughput CUDA cores. Building upon thisinsight in QoQ algorithm we introduce progressive quantization that can allowlow dequantization overhead in W4A8 GEMM. Additionally we developSmoothAttention to effectively mitigate the accuracy degradation incurred by4-bit KV quantization. In the QServe system we perform compute-aware weightreordering and take advantage of register-level parallelism to reducedequantization latency. We also make fused attention memory-bound harnessingthe performance gain brought by KV4 quantization. As a result QServe improvesthe maximum achievable serving throughput of Llama-3-8B by 1.2x on A100 1.4xon L40S and Qwen1.5-72B by 2.4x on A100 3.5x on L40S compared toTensorRT-LLM. Remarkably QServe on L40S GPU can achieve even higher throughputthan TensorRT-LLM on A100. Thus QServe effectively reduces the dollar cost ofLLM serving by 3x. Code is available at https://github.com/mit-han-lab/qserve. |


| Item |Content|
| --- |---|
|idx| 2405.04517v1 |
|title| xLSTM: Extended Long Short-Term Memory |
|authors| Maximilian BeckKorbinian PöppelMarkus SpanringAndreas AuerOleksandra PrudnikovaMichael KoppGünter KlambauerJohannes BrandstetterSepp Hochreiter
|links| http://arxiv.org/abs/2405.04517v1 |
|updated| 2024-05-07 17:50:21 UTC |
|summary| In the 1990s the constant error carousel and gating were introduced as thecentral ideas of the Long Short-Term Memory LSTM. Since then LSTMs havestood the test of time and contributed to numerous deep learning successstories in particular they constituted the first Large Language Models LLMs.However the advent of the Transformer technology with parallelizableself-attention at its core marked the dawn of a new era outpacing LSTMs atscale. We now raise a simple question: How far do we get in language modelingwhen scaling LSTMs to billions of parameters leveraging the latest techniquesfrom modern LLMs but mitigating known limitations of LSTMs Firstly weintroduce exponential gating with appropriate normalization and stabilizationtechniques. Secondly we modify the LSTM memory structure obtaining: i sLSTMwith a scalar memory a scalar update and new memory mixing ii mLSTM thatis fully parallelizable with a matrix memory and a covariance update rule.Integrating these LSTM extensions into residual block backbones yields xLSTMblocks that are then residually stacked into xLSTM architectures. Exponentialgating and modified memory structures boost xLSTM capabilities to performfavorably when compared to state-of-the-art Transformers and State SpaceModels both in performance and scaling. |


| Item |Content|
| --- |---|
|idx| 2405.04513v1 |
|title| Switchable Decision: Dynamic Neural Generation Networks |
|authors| Shujian ZhangKorawat TanwisuthChengyue GongPengcheng HeMingyuan Zhou
|links| http://arxiv.org/abs/2405.04513v1 |
|updated| 2024-05-07 17:44:54 UTC |
|summary| Auto-regressive generation models achieve competitive performance across manydifferent NLP tasks such as summarization question answering andclassifications. However they are also known for being slow in inferencewhich makes them challenging to deploy in real-time applications. We propose aswitchable decision to accelerate inference by dynamically assigningcomputation resources for each data instance. Automatically making decisions onwhere to skip and how to balance quality and computation cost with constrainedoptimization our dynamic neural generation networks enforce the efficientinference path and determine the optimized trade-off. Experiments acrossquestion answering summarization and classification benchmarks show that ourmethod benefits from less computation cost during inference while keeping thesame accuracy. Extensive experiments and ablation studies demonstrate that ourmethod can be general effective and beneficial for many NLP tasks. |


| Item |Content|
| --- |---|
|idx| 2405.04495v1 |
|title| Toward In-Context Teaching: Adapting Examples to Students' Misconceptions |
|authors| Alexis RossJacob Andreas
|links| http://arxiv.org/abs/2405.04495v1 |
|updated| 2024-05-07 17:05:27 UTC |
|summary| When a teacher provides examples for a student to study these examples mustbe informative enabling a student to progress from their current state towarda target concept or skill. Good teachers must therefore simultaneously inferwhat students already know and adapt their teaching to students changing stateof knowledge. There is increasing interest in using computational modelsparticularly large language models as pedagogical tools. As students languagemodels in particular have shown a remarkable ability to adapt to new tasksgiven small numbers of examples. But how effectively can these models adapt asteachers to students of different types To study this question we introduce asuite of models and evaluation methods we call AdapT. AdapT has two components:1 a collection of simulated Bayesian student models that can be used forevaluation of automated teaching methods 2 a platform for evaluation withhuman students to characterize the real-world effectiveness of these methods.We additionally introduce 3 AToM a new probabilistic model for adaptiveteaching that jointly infers students past beliefs and optimizes for thecorrectness of future beliefs. In evaluations of simulated students acrossthree learning domains fraction arithmetic English morphology functionlearning AToM systematically outperforms LLM-based and standard Bayesianteaching models. In human experiments both AToM and LLMs outperformnon-adaptive random example selection. Our results highlight both thedifficulty of the adaptive teaching task and the potential of learned adaptivemodels for solving it. |


| Item |Content|
| --- |---|
|idx| 2405.04491v1 |
|title| TorchDriveEnv: A Reinforcement Learning Benchmark for Autonomous Driving with Reactive, Realistic, and Diverse Non-Playable Characters |
|authors| Jonathan Wilder LavingtonKe ZhangVasileios LioutasMatthew NiedobaYunpeng LiuDylan GreenSaeid NaderipariziXiaoxuan LiangSetareh DabiriAdam ŚcibiorBerend ZwartsenbergFrank Wood
|links| http://arxiv.org/abs/2405.04491v1 |
|updated| 2024-05-07 17:02:02 UTC |
|summary| The training testing and deployment of autonomous vehicles requiresrealistic and efficient simulators. Moreover because of the high variabilitybetween different problems presented in different autonomous systems thesesimulators need to be easy to use and easy to modify. To address theseproblems we introduce TorchDriveSim and its benchmark extension TorchDriveEnv.TorchDriveEnv is a lightweight reinforcement learning benchmark programmedentirely in Python which can be modified to test a number of different factorsin learned vehicle behavior including the effect of varying kinematic modelsagent types and traffic control patterns. Most importantly unlike many replaybased simulation approaches TorchDriveEnv is fully integrated with a state ofthe art behavioral simulation API. This allows users to train and evaluatedriving models alongside data driven Non-Playable Characters NPC whoseinitializations and driving behavior are reactive realistic and diverse. Weillustrate the efficiency and simplicity of TorchDriveEnv by evaluating commonreinforcement learning baselines in both training and validation environments.Our experiments show that TorchDriveEnv is easy to use but difficult to solve. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.04533v1 |
|title| ChatHuman: Language-driven 3D Human Understanding with Retrieval-Augmented Tool Reasoning |
|authors| Jing LinYao FengWeiyang LiuMichael J. Black
|links| http://arxiv.org/abs/2405.04533v1 |
|updated| 2024-05-07 17:59:31 UTC |
|summary| Numerous methods have been proposed to detect estimate and analyzeproperties of people in images including the estimation of 3D pose shapecontact human-object interaction emotion and more. Each of these methodsworks in isolation instead of synergistically. Here we address this problem andbuild a language-driven human understanding system -- ChatHuman which combinesand integrates the skills of many different methods. To do so we finetune aLarge Language Model LLM to select and use a wide variety of existing toolsin response to user inputs. In doing so ChatHuman is able to combineinformation from multiple tools to solve problems more accurately than theindividual tools themselves and to leverage tool output to improve its abilityto reason about humans. The novel features of ChatHuman include leveragingacademic publications to guide the application of 3D human-related toolsemploying a retrieval-augmented generation model to generatein-context-learning examples for handling new tools and discriminating andintegrating tool results to enhance 3D human understanding. Our experimentsshow that ChatHuman outperforms existing models in both tool selection accuracyand performance across multiple 3D human-related tasks. ChatHuman is a steptowards consolidating diverse methods for human analysis into a singlepowerful system for 3D human reasoning. |


| Item |Content|
| --- |---|
|idx| 2405.04532v1 |
|title| QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving |
|authors| Yujun LinHaotian TangShang YangZhekai ZhangGuangxuan XiaoChuang GanSong Han
|links| http://arxiv.org/abs/2405.04532v1 |
|updated| 2024-05-07 17:59:30 UTC |
|summary| Quantization can accelerate large language model LLM inference. Goingbeyond INT8 quantization the research community is actively exploring evenlower precision such as INT4. Nonetheless state-of-the-art INT4 quantizationtechniques only accelerate low-batch edge LLM inference failing to deliverperformance gains in large-batch cloud-based LLM serving. We uncover acritical issue: existing INT4 quantization methods suffer from significantruntime overhead 20-90 when dequantizing either weights or partial sums onGPUs. To address this challenge we introduce QoQ a W4A8KV4 quantizationalgorithm with 4-bit weight 8-bit activation and 4-bit KV cache. QoQ standsfor quattuor-octo-quattuor which represents 4-8-4 in Latin. QoQ is implementedby the QServe inference library that achieves measured speedup. The key insightdriving QServe is that the efficiency of LLM serving on GPUs is criticallyinfluenced by operations on low-throughput CUDA cores. Building upon thisinsight in QoQ algorithm we introduce progressive quantization that can allowlow dequantization overhead in W4A8 GEMM. Additionally we developSmoothAttention to effectively mitigate the accuracy degradation incurred by4-bit KV quantization. In the QServe system we perform compute-aware weightreordering and take advantage of register-level parallelism to reducedequantization latency. We also make fused attention memory-bound harnessingthe performance gain brought by KV4 quantization. As a result QServe improvesthe maximum achievable serving throughput of Llama-3-8B by 1.2x on A100 1.4xon L40S and Qwen1.5-72B by 2.4x on A100 3.5x on L40S compared toTensorRT-LLM. Remarkably QServe on L40S GPU can achieve even higher throughputthan TensorRT-LLM on A100. Thus QServe effectively reduces the dollar cost ofLLM serving by 3x. Code is available at https://github.com/mit-han-lab/qserve. |


| Item |Content|
| --- |---|
|idx| 2405.04520v1 |
|title| NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts |
|authors| Shudan ZhangHanlin ZhaoXiao LiuQinkai ZhengZehan QiXiaotao GuXiaohan ZhangYuxiao DongJie Tang
|links| http://arxiv.org/abs/2405.04520v1 |
|updated| 2024-05-07 17:52:51 UTC |
|summary| Large language models LLMs have manifested strong ability to generate codesfor productive activities. However current benchmarks for code synthesis suchas HumanEval MBPP and DS-1000 are predominantly oriented towardsintroductory tasks on algorithm and data science insufficiently satisfyingchallenging requirements prevalent in real-world coding. To fill this gap wepropose NaturalCodeBench NCB a challenging code benchmark designed to mirrorthe complexity and variety of scenarios in real coding tasks. NCB comprises 402high-quality problems in Python and Java meticulously selected from naturaluser queries from online coding services covering 6 different domains. Notingthe extraordinary difficulty in creating testing cases for real-world querieswe also introduce a semi-automated pipeline to enhance the efficiency of testcase construction. Comparing with manual solutions it achieves an efficiencyincrease of more than 4 times. Our systematic experiments on 39 LLMs find thatperformance gaps on NCB between models with close HumanEval scores could stillbe significant indicating a lack of focus on practical code synthesisscenarios or over-specified optimization on HumanEval. On the other hand eventhe best-performing GPT-4 is still far from satisfying on NCB. The evaluationtoolkit and development set are available athttps://github.com/THUDM/NaturalCodeBench. |


| Item |Content|
| --- |---|
|idx| 2405.04517v1 |
|title| xLSTM: Extended Long Short-Term Memory |
|authors| Maximilian BeckKorbinian PöppelMarkus SpanringAndreas AuerOleksandra PrudnikovaMichael KoppGünter KlambauerJohannes BrandstetterSepp Hochreiter
|links| http://arxiv.org/abs/2405.04517v1 |
|updated| 2024-05-07 17:50:21 UTC |
|summary| In the 1990s the constant error carousel and gating were introduced as thecentral ideas of the Long Short-Term Memory LSTM. Since then LSTMs havestood the test of time and contributed to numerous deep learning successstories in particular they constituted the first Large Language Models LLMs.However the advent of the Transformer technology with parallelizableself-attention at its core marked the dawn of a new era outpacing LSTMs atscale. We now raise a simple question: How far do we get in language modelingwhen scaling LSTMs to billions of parameters leveraging the latest techniquesfrom modern LLMs but mitigating known limitations of LSTMs Firstly weintroduce exponential gating with appropriate normalization and stabilizationtechniques. Secondly we modify the LSTM memory structure obtaining: i sLSTMwith a scalar memory a scalar update and new memory mixing ii mLSTM thatis fully parallelizable with a matrix memory and a covariance update rule.Integrating these LSTM extensions into residual block backbones yields xLSTMblocks that are then residually stacked into xLSTM architectures. Exponentialgating and modified memory structures boost xLSTM capabilities to performfavorably when compared to state-of-the-art Transformers and State SpaceModels both in performance and scaling. |


| Item |Content|
| --- |---|
|idx| 2405.04513v1 |
|title| Switchable Decision: Dynamic Neural Generation Networks |
|authors| Shujian ZhangKorawat TanwisuthChengyue GongPengcheng HeMingyuan Zhou
|links| http://arxiv.org/abs/2405.04513v1 |
|updated| 2024-05-07 17:44:54 UTC |
|summary| Auto-regressive generation models achieve competitive performance across manydifferent NLP tasks such as summarization question answering andclassifications. However they are also known for being slow in inferencewhich makes them challenging to deploy in real-time applications. We propose aswitchable decision to accelerate inference by dynamically assigningcomputation resources for each data instance. Automatically making decisions onwhere to skip and how to balance quality and computation cost with constrainedoptimization our dynamic neural generation networks enforce the efficientinference path and determine the optimized trade-off. Experiments acrossquestion answering summarization and classification benchmarks show that ourmethod benefits from less computation cost during inference while keeping thesame accuracy. Extensive experiments and ablation studies demonstrate that ourmethod can be general effective and beneficial for many NLP tasks. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.04534v1 |
|title| Tactile-Augmented Radiance Fields |
|authors| Yiming DouFengyu YangYi LiuAntonio LoquercioAndrew Owens
|links| http://arxiv.org/abs/2405.04534v1 |
|updated| 2024-05-07 17:59:50 UTC |
|summary| We present a scene representation which we call a tactile-augmented radiancefield TaRF that brings vision and touch into a shared 3D space. Thisrepresentation can be used to estimate the visual and tactile signals for agiven 3D position within a scene. We capture a scenes TaRF from a collectionof photos and sparsely sampled touch probes. Our approach makes use of twoinsights: i common vision-based touch sensors are built on ordinary camerasand thus can be registered to images using methods from multi-view geometryand ii visually and structurally similar regions of a scene share the sametactile features. We use these insights to register touch signals to a capturedvisual scene and to train a conditional diffusion model that provided with anRGB-D image rendered from a neural radiance field generates its correspondingtactile signal. To evaluate our approach we collect a dataset of TaRFs. Thisdataset contains more touch samples than previous real-world datasets and itprovides spatially aligned visual signals for each captured touch signal. Wedemonstrate the accuracy of our cross-modal generative model and the utility ofthe captured visual-tactile data on several downstream tasks. Project page:https://dou-yiming.github.io/TaRF |


| Item |Content|
| --- |---|
|idx| 2405.04533v1 |
|title| ChatHuman: Language-driven 3D Human Understanding with Retrieval-Augmented Tool Reasoning |
|authors| Jing LinYao FengWeiyang LiuMichael J. Black
|links| http://arxiv.org/abs/2405.04533v1 |
|updated| 2024-05-07 17:59:31 UTC |
|summary| Numerous methods have been proposed to detect estimate and analyzeproperties of people in images including the estimation of 3D pose shapecontact human-object interaction emotion and more. Each of these methodsworks in isolation instead of synergistically. Here we address this problem andbuild a language-driven human understanding system -- ChatHuman which combinesand integrates the skills of many different methods. To do so we finetune aLarge Language Model LLM to select and use a wide variety of existing toolsin response to user inputs. In doing so ChatHuman is able to combineinformation from multiple tools to solve problems more accurately than theindividual tools themselves and to leverage tool output to improve its abilityto reason about humans. The novel features of ChatHuman include leveragingacademic publications to guide the application of 3D human-related toolsemploying a retrieval-augmented generation model to generatein-context-learning examples for handling new tools and discriminating andintegrating tool results to enhance 3D human understanding. Our experimentsshow that ChatHuman outperforms existing models in both tool selection accuracyand performance across multiple 3D human-related tasks. ChatHuman is a steptowards consolidating diverse methods for human analysis into a singlepowerful system for 3D human reasoning. |


| Item |Content|
| --- |---|
|idx| 2405.04496v1 |
|title| Edit-Your-Motion: Space-Time Diffusion Decoupling Learning for Video Motion Editing |
|authors| Yi ZuoLingling LiLicheng JiaoFang LiuXu LiuWenping MaShuyuan YangYuwei Guo
|links| http://arxiv.org/abs/2405.04496v1 |
|updated| 2024-05-07 17:06:59 UTC |
|summary| Existing diffusion-based video editing methods have achieved impressiveresults in motion editing. Most of the existing methods focus on the motionalignment between the edited video and the reference video. However thesemethods do not constrain the background and object content of the video toremain unchanged which makes it possible for users to generate unexpectedvideos. In this paper we propose a one-shot video motion editing method calledEdit-Your-Motion that requires only a single text-video pair for training.Specifically we design the Detailed Prompt-Guided Learning Strategy DPL todecouple spatio-temporal features in space-time diffusion models. DPL separateslearning object content and motion into two training stages. In the firsttraining stage we focus on learning the spatial features the features ofobject content and breaking down the temporal relationships in the videoframes by shuffling them. We further propose Recurrent-Causal AttentionRC-Attn to learn the consistent content features of the object from unorderedvideo frames. In the second training stage we restore the temporalrelationship in video frames to learn the temporal feature the features of thebackground and objects motion. We also adopt the Noise Constraint Loss tosmooth out inter-frame differences. Finally in the inference stage we injectthe content features of the source object into the editing branch through atwo-branch structure editing branch and reconstruction branch. WithEdit-Your-Motion users can edit the motion of objects in the source video togenerate more exciting and diverse videos. Comprehensive qualitativeexperiments quantitative experiments and user preference studies demonstratethat Edit-Your-Motion performs better than other methods. |


| Item |Content|
| --- |---|
|idx| 2405.04489v1 |
|title| S3Former: Self-supervised High-resolution Transformer for Solar PV Profiling |
|authors| Minh TranAdrian De LuisHaitao LiaoYing HuangRoy McCannAlan MantoothJack CothrenNgan Le
|links| http://arxiv.org/abs/2405.04489v1 |
|updated| 2024-05-07 16:56:21 UTC |
|summary| As the impact of climate change escalates the global necessity to transitionto sustainable energy sources becomes increasingly evident. Renewable energieshave emerged as a viable solution for users with Photovoltaic energy being afavored choice for small installations due to its reliability and efficiency.Accurate mapping of PV installations is crucial for understanding the extensionof its adoption and informing energy policy. To meet this need we introduceS3Former designed to segment solar panels from aerial imagery and provide sizeand location information critical for analyzing the impact of suchinstallations on the grid. Solar panel identification is challenging due tofactors such as varying weather conditions roof characteristics GroundSampling Distance variations and lack of appropriate initialization weights foroptimized training. To tackle these complexities S3Former features a MaskedAttention Mask Transformer incorporating a self-supervised learning pretrainedbackbone. Specifically our model leverages low-level and high-level featuresextracted from the backbone and incorporates an instance query mechanismincorporated on the Transformer architecture to enhance the localization ofsolar PV installations. We introduce a self-supervised learning phase pretexttask to improve the initialization weights on the backbone of S3Former. Weevaluated S3Former using diverse datasets demonstrate improvementstate-of-the-art models. |


| Item |Content|
| --- |---|
|idx| 2405.04459v1 |
|title| A Significantly Better Class of Activation Functions Than ReLU Like Activation Functions |
|authors| Mathew Mithra NoelYug Oswal
|links| http://arxiv.org/abs/2405.04459v1 |
|updated| 2024-05-07 16:24:03 UTC |
|summary| This paper introduces a significantly better class of activation functionsthan the almost universally used ReLU like and Sigmoidal class of activationfunctions. Two new activation functions referred to as the Cone andParabolic-Cone that differ drastically from popular activation functions andsignificantly outperform these on the CIFAR-10 and Imagenette benchmmarks areproposed. The cone activation functions are positive only on a finite intervaland are strictly negative except at the end-points of the interval where theybecome zero. Thus the set of inputs that produce a positive output for a neuronwith cone activation functions is a hyperstrip and not a half-space as is theusual case. Since a hyper strip is the region between two parallelhyper-planes it allows neurons to more finely divide the input feature spaceinto positive and negative classes than with infinitely wide half-spaces. Inparticular the XOR function can be learn by a single neuron with cone-likeactivation functions. Both the cone and parabolic-cone activation functions areshown to achieve higher accuracies with significantly fewer neurons onbenchmarks. The results presented in this paper indicate that many nonlinearreal-world datasets may be separated with fewer hyperstrips than half-spaces.The Cone and Parabolic-Cone activation functions have larger derivatives thanReLU and are shown to significantly speedup training. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.04517v1 |
|title| xLSTM: Extended Long Short-Term Memory |
|authors| Maximilian BeckKorbinian PöppelMarkus SpanringAndreas AuerOleksandra PrudnikovaMichael KoppGünter KlambauerJohannes BrandstetterSepp Hochreiter
|links| http://arxiv.org/abs/2405.04517v1 |
|updated| 2024-05-07 17:50:21 UTC |
|summary| In the 1990s the constant error carousel and gating were introduced as thecentral ideas of the Long Short-Term Memory LSTM. Since then LSTMs havestood the test of time and contributed to numerous deep learning successstories in particular they constituted the first Large Language Models LLMs.However the advent of the Transformer technology with parallelizableself-attention at its core marked the dawn of a new era outpacing LSTMs atscale. We now raise a simple question: How far do we get in language modelingwhen scaling LSTMs to billions of parameters leveraging the latest techniquesfrom modern LLMs but mitigating known limitations of LSTMs Firstly weintroduce exponential gating with appropriate normalization and stabilizationtechniques. Secondly we modify the LSTM memory structure obtaining: i sLSTMwith a scalar memory a scalar update and new memory mixing ii mLSTM thatis fully parallelizable with a matrix memory and a covariance update rule.Integrating these LSTM extensions into residual block backbones yields xLSTMblocks that are then residually stacked into xLSTM architectures. Exponentialgating and modified memory structures boost xLSTM capabilities to performfavorably when compared to state-of-the-art Transformers and State SpaceModels both in performance and scaling. |


| Item |Content|
| --- |---|
|idx| 2405.04393v1 |
|title| Efficient Online Set-valued Classification with Bandit Feedback |
|authors| Zhou WangXingye Qiao
|links| http://arxiv.org/abs/2405.04393v1 |
|updated| 2024-05-07 15:14:51 UTC |
|summary| Conformal prediction is a distribution-free method that wraps a given machinelearning model and returns a set of plausible labels that contain the truelabel with a prescribed coverage rate. In practice the empirical coverageachieved highly relies on fully observed label information from data both inthe training phase for model fitting and the calibration phase for quantileestimation. This dependency poses a challenge in the context of online learningwith bandit feedback where a learner only has access to the correctness ofactions i.e. pulled an arm but not the full information of the true label.In particular when the pulled arm is incorrect the learner only knows thatthe pulled one is not the true class label but does not know which label istrue. Additionally bandit feedback further results in a smaller labeleddataset for calibration limited to instances with correct actions therebyaffecting the accuracy of quantile estimation. To address these limitations wepropose Bandit Class-specific Conformal Prediction BCCP offering coverageguarantees on a class-specific granularity. Using an unbiased estimation of anestimand involving the true label BCCP trains the model and makes set-valuedinferences through stochastic gradient descent. Our approach overcomes thechallenges of sparsely labeled data in each iteration and generalizes thereliability and applicability of conformal prediction to online decision-makingenvironments. |


| Item |Content|
| --- |---|
|idx| 2405.04346v1 |
|title| Revisiting character-level adversarial attacks |
|authors| Elias Abad RocamoraYongtao WuFanghui LiuGrigorios G. ChrysosVolkan Cevher
|links| http://arxiv.org/abs/2405.04346v1 |
|updated| 2024-05-07 14:23:22 UTC |
|summary| Adversarial attacks in Natural Language Processing apply perturbations in thecharacter or token levels. Token-level attacks gaining prominence for theiruse of gradient-based methods are susceptible to altering sentence semanticsleading to invalid adversarial examples. While character-level attacks easilymaintain semantics they have received less attention as they cannot easilyadopt popular gradient-based methods and are thought to be easy to defend.Challenging these beliefs we introduce Charmer an efficient query-basedadversarial attack capable of achieving high attack success rate ASR whilegenerating highly similar adversarial examples. Our method successfully targetsboth small BERT and large Llama 2 models. Specifically on BERT with SST-2Charmer improves the ASR in 4.84 points and the USE similarity in 8 pointswith respect to the previous art. Our implementation is available inhttps://github.com/LIONS-EPFL/Charmer. |


| Item |Content|
| --- |---|
|idx| 2405.04147v1 |
|title| Multiparameter regularization and aggregation in the context of polynomial functional regression |
|authors| Elke R. GizewskiMarkus HolzleitnerLukas Mayer-SuessSergiy Pereverzyev Jr.Sergei V. Pereverzyev
|links| http://arxiv.org/abs/2405.04147v1 |
|updated| 2024-05-07 09:26:20 UTC |
|summary| Most of the recent results in polynomial functional regression have beenfocused on an in-depth exploration of single-parameter regularization schemes.In contrast in this study we go beyond that framework by introducing analgorithm for multiple parameter regularization and presenting a theoreticallygrounded method for dealing with the associated parameters. This methodfacilitates the aggregation of models with varying regularization parameters.The efficacy of the proposed approach is assessed through evaluations on bothsynthetic and some real-world medical data revealing promising results. |


| Item |Content|
| --- |---|
|idx| 2405.04043v1 |
|title| Scalable Vertical Federated Learning via Data Augmentation and Amortized Inference |
|authors| Conor HassanMatthew SuttonAntonietta MiraKerrie Mengersen
|links| http://arxiv.org/abs/2405.04043v1 |
|updated| 2024-05-07 06:29:06 UTC |
|summary| Vertical federated learning VFL has emerged as a paradigm for collaborativemodel estimation across multiple clients each holding a distinct set ofcovariates. This paper introduces the first comprehensive framework for fittingBayesian models in the VFL setting. We propose a novel approach that leveragesdata augmentation techniques to transform VFL problems into a form compatiblewith existing Bayesian federated learning algorithms. We present an innovativemodel formulation for specific VFL scenarios where the joint likelihoodfactorizes into a product of client-specific likelihoods. To mitigate thedimensionality challenge posed by data augmentation which scales with thenumber of observations and clients we develop a factorized amortizedvariational approximation that achieves scalability independent of the numberof observations. We showcase the efficacy of our framework through extensivenumerical experiments on logistic regression multilevel regression and anovel hierarchical Bayesian split neural net model. Our work paves the way forprivacy-preserving decentralized Bayesian inference in vertically partitioneddata scenarios opening up new avenues for research and applications in variousdomains. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.04497v1 |
|title| Unveiling Disparities in Web Task Handling Between Human and Web Agent |
|authors| Kihoon SonJinhyeon KwonDeEun ChoiTae Soo KimYoung-Ho KimSangdoo YunJuho Kim
|links| http://arxiv.org/abs/2405.04497v1 |
|updated| 2024-05-07 17:10:31 UTC |
|summary| With the advancement of Large-Language Models LLMs and LargeVision-Language Models LVMs agents have shown significant capabilities invarious tasks such as data analysis gaming or code generation. Recentlythere has been a surge in research on web agents capable of performing taskswithin the web environment. However the web poses unforeseeable scenarioschallenging the generalizability of these agents. This study investigates thedisparities between human and web agents performance in web tasks e.g.information search by concentrating on planning action and reflectionaspects during task execution. We conducted a web task study with a think-aloudprotocol revealing distinct cognitive actions and operations on websitesemployed by humans. Comparative examination of existing agent structures andhuman behavior with thought processes highlighted differences in knowledgeupdating and ambiguity handling when performing the task. Humans demonstrated apropensity for exploring and modifying plans based on additional informationand investigating reasons for failure. These findings offer insights intodesigning planning reflection and information discovery modules for webagents and designing the capturing method for implicit human knowledge in a webtask. |


| Item |Content|
| --- |---|
|idx| 2405.04457v1 |
|title| Towards Geographic Inclusion in the Evaluation of Text-to-Image Models |
|authors| Melissa HallSamuel J. BellCandace RossAdina WilliamsMichal DrozdzalAdriana Romero Soriano
|links| http://arxiv.org/abs/2405.04457v1 |
|updated| 2024-05-07 16:23:06 UTC |
|summary| Rapid progress in text-to-image generative models coupled with theirdeployment for visual content creation has magnified the importance ofthoroughly evaluating their performance and identifying potential biases. Inpursuit of models that generate images that are realistic diverse visuallyappealing and consistent with the given prompt researchers and practitionersoften turn to automated metrics to facilitate scalable and cost-effectiveperformance profiling. However commonly-used metrics often fail to account forthe full diversity of human preference often even in-depth human evaluationsface challenges with subjectivity especially as interpretations of evaluationcriteria vary across regions and cultures. In this work we conduct a largecross-cultural study to study how much annotators in Africa Europe andSoutheast Asia vary in their perception of geographic representation visualappeal and consistency in real and generated images from state-of-the artpublic APIs. We collect over 65000 image annotations and 20 survey responses.We contrast human annotations with common automated metrics finding that humanpreferences vary notably across geographic location and that current metrics donot fully account for this diversity. For example annotators in differentlocations often disagree on whether exaggerated stereotypical depictions of aregion are considered geographically representative. In addition the utilityof automatic evaluations is dependent on assumptions about their set-up suchas the alignment of feature extractors with human perception of objectsimilarity or the definition of appeal captured in reference datasets used toground evaluations. We recommend steps for improved automatic and humanevaluations. |


| Item |Content|
| --- |---|
|idx| 2405.04382v1 |
|title| Large Language Models Cannot Explain Themselves |
|authors| Advait Sarkar
|links| http://arxiv.org/abs/2405.04382v1 |
|updated| 2024-05-07 15:05:23 UTC |
|summary| Large language models can be prompted to produce text. They can also beprompted to produce explanations of their output. But these are not reallyexplanations because they do not accurately reflect the mechanical processunderlying the prediction. The illusion that they reflect the reasoning processcan result in significant harms. These explanations can be valuable but forpromoting critical thinking rather than for understanding the model. I proposea recontextualisation of these explanations using the term exoplanationsto draw attention to their exogenous nature. I discuss some implications fordesign and technology such as the inclusion of appropriate guardrails andresponses when models are prompted to generate explanations. |


| Item |Content|
| --- |---|
|idx| 2405.04251v1 |
|title| A General Model for Detecting Learner Engagement: Implementation and Evaluation |
|authors| Somayeh MalekshahiJavad M. KheyridoostOmid Fatemi
|links| http://arxiv.org/abs/2405.04251v1 |
|updated| 2024-05-07 12:11:15 UTC |
|summary| Considering learner engagement has a mutual benefit for both learners andinstructors. Instructors can help learners increase their attentioninvolvement motivation and interest. On the other hand instructors canimprove their instructional performance by evaluating the cumulative results ofall learners and upgrading their training programs. This paper proposes ageneral lightweight model for selecting and processing features to detectlearners engagement levels while preserving the sequential temporalrelationship over time. During training and testing we analyzed the videosfrom the publicly available DAiSEE dataset to capture the dynamic essence oflearner engagement. We have also proposed an adaptation policy to find newlabels that utilize the affective states of this dataset related to educationthereby improving the models judgment. The suggested model achieves anaccuracy of 68.57 in a specific implementation and outperforms the studiedstate-of-the-art models detecting learners engagement levels. |


| Item |Content|
| --- |---|
|idx| 2405.04054v1 |
|title| What Impacts the Quality of the User Answers when Asked about the Current Context? |
|authors| Ivano BisonHaonan ZhaoFausto Giunchiglia
|links| http://arxiv.org/abs/2405.04054v1 |
|updated| 2024-05-07 06:55:10 UTC |
|summary| Sensor data provide an objective view of reality but fail to capture thesubjective motivations behind an individuals behavior. This latter informationis crucial for learning about the various dimensions of the personal contextthus increasing predictability. The main limitation is the human input whichis often not of the quality that is needed. The work so far has focused on theusually high number of missing answers. The focus of this paper is ontextitthe number of mistakes made when answering questions. Three are themain contributions of this paper. First we show that the users reaction timei.e. the time before starting to respond is the main cause of a low answerquality where its effects are both direct and indirect the latter relating toits impact on the completion time i.e. the time taken to compile theresponse. Second we identify the specific exogenous e.g. the situational ortemporal context and endogenous e.g. mood personality traits factors whichhave an influence on the reaction time as well as on the completion time.Third we show how reaction and completion time compose their effects on theanswer quality. The paper concludes with a set of actionable recommendations. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.04491v1 |
|title| TorchDriveEnv: A Reinforcement Learning Benchmark for Autonomous Driving with Reactive, Realistic, and Diverse Non-Playable Characters |
|authors| Jonathan Wilder LavingtonKe ZhangVasileios LioutasMatthew NiedobaYunpeng LiuDylan GreenSaeid NaderipariziXiaoxuan LiangSetareh DabiriAdam ŚcibiorBerend ZwartsenbergFrank Wood
|links| http://arxiv.org/abs/2405.04491v1 |
|updated| 2024-05-07 17:02:02 UTC |
|summary| The training testing and deployment of autonomous vehicles requiresrealistic and efficient simulators. Moreover because of the high variabilitybetween different problems presented in different autonomous systems thesesimulators need to be easy to use and easy to modify. To address theseproblems we introduce TorchDriveSim and its benchmark extension TorchDriveEnv.TorchDriveEnv is a lightweight reinforcement learning benchmark programmedentirely in Python which can be modified to test a number of different factorsin learned vehicle behavior including the effect of varying kinematic modelsagent types and traffic control patterns. Most importantly unlike many replaybased simulation approaches TorchDriveEnv is fully integrated with a state ofthe art behavioral simulation API. This allows users to train and evaluatedriving models alongside data driven Non-Playable Characters NPC whoseinitializations and driving behavior are reactive realistic and diverse. Weillustrate the efficiency and simplicity of TorchDriveEnv by evaluating commonreinforcement learning baselines in both training and validation environments.Our experiments show that TorchDriveEnv is easy to use but difficult to solve. |


| Item |Content|
| --- |---|
|idx| 2405.04219v1 |
|title| Iterative Experience Refinement of Software-Developing Agents |
|authors| Chen QianJiahao LiYufan DangWei LiuYiFei WangZihao XieWeize ChenCheng YangYingli ZhangZhiyuan LiuMaosong Sun
|links| http://arxiv.org/abs/2405.04219v1 |
|updated| 2024-05-07 11:33:49 UTC |
|summary| Autonomous agents powered by large language models LLMs show significantpotential for achieving high autonomy in various scenarios such as softwaredevelopment. Recent research has shown that LLM agents can leverage pastexperiences to reduce errors and enhance efficiency. However the staticexperience paradigm reliant on a fixed collection of past experiences acquiredheuristically lacks iterative refinement and thus hampers agentsadaptability. In this paper we introduce the Iterative Experience Refinementframework enabling LLM agents to refine experiences iteratively during taskexecution. We propose two fundamental patterns: the successive patternrefining based on nearest experiences within a task batch and the cumulativepattern acquiring experiences across all previous task batches. Augmented withour heuristic experience elimination the method prioritizes high-quality andfrequently-used experiences effectively managing the experience space andenhancing efficiency. Extensive experiments show that while the successivepattern may yield superior results the cumulative pattern provides more stableperformance. Moreover experience elimination facilitates achieving betterperformance using just 11.54 of a high-quality subset. |


| Item |Content|
| --- |---|
|idx| 2405.03994v1 |
|title| A Guide to Re-Implementing Agent-based Models: Experiences from the HUMAT Model |
|authors| Önder GürcanTimo SzczepanskaPatrycja Antosz
|links| http://arxiv.org/abs/2405.03994v1 |
|updated| 2024-05-07 04:17:03 UTC |
|summary| Replicating existing agent-based models poses significant challengesparticularly for those new to the field. This article presents an all-encompassing guide to re-implementing agent-based models encompassing vitalconcepts such as comprehending the original model utilizing agent-basedmodeling frameworks simulation design model validation and more. Byembracing the proposed guide researchers and practitioners can gain a profoundunderstanding of the entire re-implementation process resulting in heightenedaccuracy and reliability of simulations for complex systems. Furthermore thisarticle showcases the re-implementation of the HUMAT socio-cognitivearchitecture with a specific focus on designing a versatilelanguage-independent model. The encountered challenges and pitfalls in there-implementation process are thoroughly discussed empowering readers withpractical insights. Embrace this guide to expedite model development whileensuring robust and precise simulations. |


| Item |Content|
| --- |---|
|idx| 2405.03971v1 |
|title| Unified End-to-End V2X Cooperative Autonomous Driving |
|authors| Zhiwei LiBozhen ZhangLei YangTianyu ShenNuo XuRuosen HaoWeiting LiTao YanHuaping Liu
|links| http://arxiv.org/abs/2405.03971v1 |
|updated| 2024-05-07 03:01:40 UTC |
|summary| V2X cooperation through the integration of sensor data from both vehiclesand infrastructure is considered a pivotal approach to advancing autonomousdriving technology. Current research primarily focuses on enhancing perceptionaccuracy often overlooking the systematic improvement of accident predictionaccuracy through end-to-end learning leading to insufficient attention to thesafety issues of autonomous driving. To address this challenge this paperintroduces the UniE2EV2X framework a V2X-integrated end-to-end autonomousdriving system that consolidates key driving modules within a unified network.The framework employs a deformable attention-based data fusion strategyeffectively facilitating cooperation between vehicles and infrastructure. Themain advantages include: 1 significantly enhancing agents perception andmotion prediction capabilities thereby improving the accuracy of accidentpredictions 2 ensuring high reliability in the data fusion process 3superior end-to-end perception compared to modular approaches. Furthermore Weimplement the UniE2EV2X framework on the challenging DeepAccident a simulationdataset designed for V2X cooperative driving. |


| Item |Content|
| --- |---|
|idx| 2405.03735v1 |
|title| Select to Perfect: Imitating desired behavior from large multi-agent data |
|authors| Tim FranzmeyerEdith ElkindPhilip TorrJakob FoersterJoao Henriques
|links| http://arxiv.org/abs/2405.03735v1 |
|updated| 2024-05-06 15:48:24 UTC |
|summary| AI agents are commonly trained with large datasets of demonstrations of humanbehavior. However not all behaviors are equally safe or desirable. Desiredcharacteristics for an AI agent can be expressed by assigning desirabilityscores which we assume are not assigned to individual behaviors but tocollective trajectories. For example in a dataset of vehicle interactionsthese scores might relate to the number of incidents that occurred. We firstassess the effect of each individual agents behavior on the collectivedesirability score e.g. assessing how likely an agent is to cause incidents.This allows us to selectively imitate agents with a positive effect e.g. onlyimitating agents that are unlikely to cause incidents. To enable this wepropose the concept of an agents Exchange Value which quantifies anindividual agents contribution to the collective desirability score. TheExchange Value is the expected change in desirability score when substitutingthe agent for a randomly selected agent. We propose additional methods forestimating Exchange Values from real-world datasets enabling us to learndesired imitation policies that outperform relevant baselines. The projectwebsite can be found at https://tinyurl.com/select-to-perfect. |


