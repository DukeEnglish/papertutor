# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.04632v1 |
|title| Arctic-TILT. Business Document Understanding at Sub-Billion Scale |
|authors| Łukasz BorchmannMichał PietruszkaWojciech JaśkowskiDawid JurkiewiczPiotr HalamaPaweł JóziakŁukasz GarncarekPaweł LiskowskiKarolina SzyndlerAndrzej GretkowskiJulita OłtusekGabriela NowakowskaArtur ZawłockiŁukasz DuhrPaweł DydaMichał Turski
|links| http://arxiv.org/abs/2408.04632v1 |
|updated| 2024-08-08 17:59:46 UTC |
|summary| The vast portion of workloads employing LLMs involves answering questionsgrounded on PDF or scan content. We introduce the Arctic-TILT achievingaccuracy on par with models 1000times its size on these use cases. It can befine-tuned and deployed on a single 24GB GPU lowering operational costs whileprocessing Visually Rich Documents with up to 400k tokens. The modelestablishes state-of-the-art results on seven diverse Document Understandingbenchmarks as well as provides reliable confidence scores and quick inferencewhich are essential for processing files in large-scale or time-sensitiveenterprise environments. |


| Item |Content|
| --- |---|
|idx| 2408.04628v1 |
|title| LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP |
|authors| Danlu ChenFreda ShiAditi AgarwalJacobo MyerstonTaylor Berg-Kirkpatrick
|links| http://arxiv.org/abs/2408.04628v1 |
|updated| 2024-08-08 17:58:06 UTC |
|summary| Standard natural language processing NLP pipelines operate on symbolicrepresentations of language which typically consist of sequences of discretetokens. However creating an analogous representation for ancient logographicwriting systems is an extremely labor intensive process that requires expertknowledge. At present a large portion of logographic data persists in a purelyvisual form due to the absence of transcription -- this issue poses abottleneck for researchers seeking to apply NLP toolkits to study ancientlogographic languages: most of the relevant data are images of writing.  This paper investigates whether direct processing of visual representationsof language offers a potential solution. We introduce LogogramNLP the firstbenchmark enabling NLP analysis of ancient logographic languages featuringboth transcribed and visual datasets for four writing systems along withannotations for tasks like classification translation and parsing. Ourexperiments compare systems that employ recent visual and text encodingstrategies as backbones. The results demonstrate that visual representationsoutperform textual representations for some investigated tasks suggesting thatvisual processing pipelines may unlock a large amount of cultural heritage dataof logographic languages for NLP-based analyses. |


| Item |Content|
| --- |---|
|idx| 2408.04619v1 |
|title| Transformer Explainer: Interactive Learning of Text-Generative Models |
|authors| Aeree ChoGrace C. KimAlexander KarpekovAlec HelblingZijie J. WangSeongmin LeeBenjamin HooverDuen Horng Chau
|links| http://arxiv.org/abs/2408.04619v1 |
|updated| 2024-08-08 17:49:07 UTC |
|summary| Transformers have revolutionized machine learning yet their inner workingsremain opaque to many. We present Transformer Explainer an interactivevisualization tool designed for non-experts to learn about Transformers throughthe GPT-2 model. Our tool helps users understand complex Transformer conceptsby integrating a model overview and enabling smooth transitions acrossabstraction levels of mathematical operations and model structures. It runs alive GPT-2 instance locally in the users browser empowering users toexperiment with their own input and observe in real-time how the internalcomponents and parameters of the Transformer work together to predict the nexttokens. Our tool requires no installation or special hardware broadening thepublics education access to modern generative AI techniques. Our open-sourcedtool is available at https://poloclub.github.io/transformer-explainer/. A videodemo is available at https://youtu.be/ECR4oAwocjs. |


| Item |Content|
| --- |---|
|idx| 2408.04614v1 |
|title| Better Alignment with Instruction Back-and-Forth Translation |
|authors| Thao NguyenJeffrey LiSewoong OhLudwig SchmidtJason WestonLuke ZettlemoyerXian Li
|links| http://arxiv.org/abs/2408.04614v1 |
|updated| 2024-08-08 17:42:32 UTC |
|summary| We propose a new method instruction back-and-forth translation to constructhigh-quality synthetic data grounded in world knowledge for aligning largelanguage models LLMs. Given documents from a web corpus we generate andcurate synthetic instructions using the backtranslation approach proposed by Liet al.2023a and rewrite the responses to improve their quality further basedon the initial documents. Fine-tuning with the resulting backtranslatedinstruction rewritten response pairs yields higher win rates on AlpacaEvalthan using other common instruction datasets such as Humpback ShareGPT OpenOrca Alpaca-GPT4 and Self-instruct. We also demonstrate that rewriting theresponses with an LLM outperforms direct distillation and the two generatedtext distributions exhibit significant distinction in embedding space. Furtheranalysis shows that our backtranslated instructions are of higher quality thanother sources of synthetic instructions while our responses are more diverseand complex than those obtained from distillation. Overall we find thatinstruction back-and-forth translation combines the best of both worlds --making use of the information diversity and quantity found on the web whileensuring the quality of the responses which is necessary for effectivealignment. |


| Item |Content|
| --- |---|
|idx| 2408.04596v1 |
|title| Code-switching in text and speech reveals information-theoretic audience design |
|authors| Debasmita BhattacharyaMarten van Schijndel
|links| http://arxiv.org/abs/2408.04596v1 |
|updated| 2024-08-08 17:14:12 UTC |
|summary| In this work we use language modeling to investigate the factors thatinfluence code-switching. Code-switching occurs when a speaker alternatesbetween one language variety the primary language and another the secondarylanguage and is widely observed in multilingual contexts. Recent work hasshown that code-switching is often correlated with areas of high informationload in the primary language but it is unclear whether high primary languageload only makes the secondary language relatively easier to produce atcode-switching points speaker-driven code-switching or whethercode-switching is additionally used by speakers to signal the need for greaterattention on the part of listeners audience-driven code-switching. In thispaper we use bilingual Chinese-English online forum posts and transcripts ofspontaneous Chinese-English speech to replicate prior findings that highprimary language Chinese information load is correlated with switches to thesecondary language English. We then demonstrate that the information load ofthe English productions is even higher than that of meaning equivalent Chinesealternatives and these are therefore not easier to produce providing evidenceof audience-driven influences in code-switching at the level of thecommunication channel not just at the sociolinguistic level in both writingand speech. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.04631v1 |
|title| Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics |
|authors| Ruining LiChuanxia ZhengChristian RupprechtAndrea Vedaldi
|links| http://arxiv.org/abs/2408.04631v1 |
|updated| 2024-08-08 17:59:38 UTC |
|summary| We present Puppet-Master an interactive video generative model that canserve as a motion prior for part-level dynamics. At test time given a singleimage and a sparse set of motion trajectories i.e. drags Puppet-Master cansynthesize a video depicting realistic part-level motion faithful to the givendrag interactions. This is achieved by fine-tuning a large-scale pre-trainedvideo diffusion model for which we propose a new conditioning architecture toinject the dragging control effectively. More importantly we introduce theall-to-first attention mechanism a drop-in replacement for the widely adoptedspatial attention modules which significantly improves generation quality byaddressing the appearance and background issues in existing models. Unlikeother motion-conditioned video generators that are trained on in-the-wildvideos and mostly move an entire object Puppet-Master is learned fromObjaverse-Animation-HQ a new dataset of curated part-level motion clips. Wepropose a strategy to automatically filter out sub-optimal animations andaugment the synthetic renderings with meaningful motion trajectories.Puppet-Master generalizes well to real images across various categories andoutperforms existing methods in a zero-shot manner on a real-world benchmark.See our project page for more results: vgg-puppetmaster.github.io. |


| Item |Content|
| --- |---|
|idx| 2408.04628v1 |
|title| LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP |
|authors| Danlu ChenFreda ShiAditi AgarwalJacobo MyerstonTaylor Berg-Kirkpatrick
|links| http://arxiv.org/abs/2408.04628v1 |
|updated| 2024-08-08 17:58:06 UTC |
|summary| Standard natural language processing NLP pipelines operate on symbolicrepresentations of language which typically consist of sequences of discretetokens. However creating an analogous representation for ancient logographicwriting systems is an extremely labor intensive process that requires expertknowledge. At present a large portion of logographic data persists in a purelyvisual form due to the absence of transcription -- this issue poses abottleneck for researchers seeking to apply NLP toolkits to study ancientlogographic languages: most of the relevant data are images of writing.  This paper investigates whether direct processing of visual representationsof language offers a potential solution. We introduce LogogramNLP the firstbenchmark enabling NLP analysis of ancient logographic languages featuringboth transcribed and visual datasets for four writing systems along withannotations for tasks like classification translation and parsing. Ourexperiments compare systems that employ recent visual and text encodingstrategies as backbones. The results demonstrate that visual representationsoutperform textual representations for some investigated tasks suggesting thatvisual processing pipelines may unlock a large amount of cultural heritage dataof logographic languages for NLP-based analyses. |


| Item |Content|
| --- |---|
|idx| 2408.04619v1 |
|title| Transformer Explainer: Interactive Learning of Text-Generative Models |
|authors| Aeree ChoGrace C. KimAlexander KarpekovAlec HelblingZijie J. WangSeongmin LeeBenjamin HooverDuen Horng Chau
|links| http://arxiv.org/abs/2408.04619v1 |
|updated| 2024-08-08 17:49:07 UTC |
|summary| Transformers have revolutionized machine learning yet their inner workingsremain opaque to many. We present Transformer Explainer an interactivevisualization tool designed for non-experts to learn about Transformers throughthe GPT-2 model. Our tool helps users understand complex Transformer conceptsby integrating a model overview and enabling smooth transitions acrossabstraction levels of mathematical operations and model structures. It runs alive GPT-2 instance locally in the users browser empowering users toexperiment with their own input and observe in real-time how the internalcomponents and parameters of the Transformer work together to predict the nexttokens. Our tool requires no installation or special hardware broadening thepublics education access to modern generative AI techniques. Our open-sourcedtool is available at https://poloclub.github.io/transformer-explainer/. A videodemo is available at https://youtu.be/ECR4oAwocjs. |


| Item |Content|
| --- |---|
|idx| 2408.04614v1 |
|title| Better Alignment with Instruction Back-and-Forth Translation |
|authors| Thao NguyenJeffrey LiSewoong OhLudwig SchmidtJason WestonLuke ZettlemoyerXian Li
|links| http://arxiv.org/abs/2408.04614v1 |
|updated| 2024-08-08 17:42:32 UTC |
|summary| We propose a new method instruction back-and-forth translation to constructhigh-quality synthetic data grounded in world knowledge for aligning largelanguage models LLMs. Given documents from a web corpus we generate andcurate synthetic instructions using the backtranslation approach proposed by Liet al.2023a and rewrite the responses to improve their quality further basedon the initial documents. Fine-tuning with the resulting backtranslatedinstruction rewritten response pairs yields higher win rates on AlpacaEvalthan using other common instruction datasets such as Humpback ShareGPT OpenOrca Alpaca-GPT4 and Self-instruct. We also demonstrate that rewriting theresponses with an LLM outperforms direct distillation and the two generatedtext distributions exhibit significant distinction in embedding space. Furtheranalysis shows that our backtranslated instructions are of higher quality thanother sources of synthetic instructions while our responses are more diverseand complex than those obtained from distillation. Overall we find thatinstruction back-and-forth translation combines the best of both worlds --making use of the information diversity and quantity found on the web whileensuring the quality of the responses which is necessary for effectivealignment. |


| Item |Content|
| --- |---|
|idx| 2408.04595v1 |
|title| Inference with the Upper Confidence Bound Algorithm |
|authors| Koulik KhamaruCun-Hui Zhang
|links| http://arxiv.org/abs/2408.04595v1 |
|updated| 2024-08-08 17:11:36 UTC |
|summary| In this paper we discuss the asymptotic behavior of the Upper ConfidenceBound UCB algorithm in the context of multiarmed bandit problems and discussits implication in downstream inferential tasks. While inferential tasks becomechallenging when data is collected in a sequential manner we argue that thisproblem can be alleviated when the sequential algorithm at hand satisfiescertain stability property. This notion of stability is motivated from theseminal work of Lai and Wei 1982. Our first main result shows that such astability property is always satisfied for the UCB algorithm and as a resultthe sample means for each arm are asymptotically normal. Next we examine thestability properties of the UCB algorithm when the number of arms K isallowed to grow with the number of arm pulls T. We show that in such a casethe arms are stable when fraclog Klog T rightarrow 0 and the numberof near-optimal arms are large. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.04619v1 |
|title| Transformer Explainer: Interactive Learning of Text-Generative Models |
|authors| Aeree ChoGrace C. KimAlexander KarpekovAlec HelblingZijie J. WangSeongmin LeeBenjamin HooverDuen Horng Chau
|links| http://arxiv.org/abs/2408.04619v1 |
|updated| 2024-08-08 17:49:07 UTC |
|summary| Transformers have revolutionized machine learning yet their inner workingsremain opaque to many. We present Transformer Explainer an interactivevisualization tool designed for non-experts to learn about Transformers throughthe GPT-2 model. Our tool helps users understand complex Transformer conceptsby integrating a model overview and enabling smooth transitions acrossabstraction levels of mathematical operations and model structures. It runs alive GPT-2 instance locally in the users browser empowering users toexperiment with their own input and observe in real-time how the internalcomponents and parameters of the Transformer work together to predict the nexttokens. Our tool requires no installation or special hardware broadening thepublics education access to modern generative AI techniques. Our open-sourcedtool is available at https://poloclub.github.io/transformer-explainer/. A videodemo is available at https://youtu.be/ECR4oAwocjs. |


| Item |Content|
| --- |---|
|idx| 2408.04614v1 |
|title| Better Alignment with Instruction Back-and-Forth Translation |
|authors| Thao NguyenJeffrey LiSewoong OhLudwig SchmidtJason WestonLuke ZettlemoyerXian Li
|links| http://arxiv.org/abs/2408.04614v1 |
|updated| 2024-08-08 17:42:32 UTC |
|summary| We propose a new method instruction back-and-forth translation to constructhigh-quality synthetic data grounded in world knowledge for aligning largelanguage models LLMs. Given documents from a web corpus we generate andcurate synthetic instructions using the backtranslation approach proposed by Liet al.2023a and rewrite the responses to improve their quality further basedon the initial documents. Fine-tuning with the resulting backtranslatedinstruction rewritten response pairs yields higher win rates on AlpacaEvalthan using other common instruction datasets such as Humpback ShareGPT OpenOrca Alpaca-GPT4 and Self-instruct. We also demonstrate that rewriting theresponses with an LLM outperforms direct distillation and the two generatedtext distributions exhibit significant distinction in embedding space. Furtheranalysis shows that our backtranslated instructions are of higher quality thanother sources of synthetic instructions while our responses are more diverseand complex than those obtained from distillation. Overall we find thatinstruction back-and-forth translation combines the best of both worlds --making use of the information diversity and quantity found on the web whileensuring the quality of the responses which is necessary for effectivealignment. |


| Item |Content|
| --- |---|
|idx| 2408.04607v1 |
|title| Risk and cross validation in ridge regression with correlated samples |
|authors| Alexander AtanasovJacob A. Zavatone-VethCengiz Pehlevan
|links| http://arxiv.org/abs/2408.04607v1 |
|updated| 2024-08-08 17:27:29 UTC |
|summary| Recent years have seen substantial advances in our understanding ofhigh-dimensional ridge regression but existing theories assume that trainingexamples are independent. By leveraging recent techniques from random matrixtheory and free probability we provide sharp asymptotics for the in- andout-of-sample risks of ridge regression when the data points have arbitrarycorrelations. We demonstrate that in this setting the generalized crossvalidation estimator GCV fails to correctly predict the out-of-sample risk.However in the case where the noise residuals have the same correlations asthe data points one can modify the GCV to yield an efficiently-computableunbiased estimator that concentrates in the high-dimensional limit which wedub CorrGCV. We further extend our asymptotic analysis to the case where thetest point has nontrivial correlations with the training set a setting oftenencountered in time series forecasting. Assuming knowledge of the correlationstructure of the time series this again yields an extension of the GCVestimator and sharply characterizes the degree to which such test points yieldan overly optimistic prediction of long-time risk. We validate the predictionsof our theory across a variety of high dimensional data. |


| Item |Content|
| --- |---|
|idx| 2408.04595v1 |
|title| Inference with the Upper Confidence Bound Algorithm |
|authors| Koulik KhamaruCun-Hui Zhang
|links| http://arxiv.org/abs/2408.04595v1 |
|updated| 2024-08-08 17:11:36 UTC |
|summary| In this paper we discuss the asymptotic behavior of the Upper ConfidenceBound UCB algorithm in the context of multiarmed bandit problems and discussits implication in downstream inferential tasks. While inferential tasks becomechallenging when data is collected in a sequential manner we argue that thisproblem can be alleviated when the sequential algorithm at hand satisfiescertain stability property. This notion of stability is motivated from theseminal work of Lai and Wei 1982. Our first main result shows that such astability property is always satisfied for the UCB algorithm and as a resultthe sample means for each arm are asymptotically normal. Next we examine thestability properties of the UCB algorithm when the number of arms K isallowed to grow with the number of arm pulls T. We show that in such a casethe arms are stable when fraclog Klog T rightarrow 0 and the numberof near-optimal arms are large. |


| Item |Content|
| --- |---|
|idx| 2408.04590v1 |
|title| Learn To Learn More Precisely |
|authors| Runxi ChengYongxian WeiXianglong HeWanyun ZhuSongsong HuangFei Richard YuFei MaChun Yuan
|links| http://arxiv.org/abs/2408.04590v1 |
|updated| 2024-08-08 17:01:26 UTC |
|summary| Meta-learning has been extensively applied in the domains of few-shotlearning and fast adaptation achieving remarkable performance. WhileMeta-learning methods like Model-Agnostic Meta-Learning MAML and its variantsprovide a good set of initial parameters for the model the model still tendsto learn shortcut features which leads to poor generalization. In this paperwe propose the formal conception of learn to learn more precisely which aimsto make the model learn precise target knowledge from data and reduce theeffect of noisy knowledge such as background and noise. To achieve thistarget we proposed a simple and effective meta-learning framework named MetaSelf-DistillationMSD to maximize the consistency of learned knowledgeenhancing the models ability to learn precise target knowledge. In the innerloop MSD uses different augmented views of the same support data to update themodel respectively. Then in the outer loop MSD utilizes the same query data tooptimize the consistency of learned knowledge enhancing the models ability tolearn more precisely. Our experiment demonstrates that MSD exhibits remarkableperformance in few-shot classification tasks in both standard and augmentedscenarios effectively boosting the accuracy and consistency of knowledgelearned by the model. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.04633v1 |
|title| LiDAR-Event Stereo Fusion with Hallucinations |
|authors| Luca BartolomeiMatteo PoggiAndrea ContiStefano Mattoccia
|links| http://arxiv.org/abs/2408.04633v1 |
|updated| 2024-08-08 17:59:58 UTC |
|summary| Event stereo matching is an emerging technique to estimate depth fromneuromorphic cameras however events are unlikely to trigger in the absence ofmotion or the presence of large untextured regions making the correspondenceproblem extremely challenging. Purposely we propose integrating a stereo eventcamera with a fixed-frequency active sensor -- e.g. a LiDAR -- collectingsparse depth measurements overcoming the aforementioned limitations. Suchdepth hints are used by hallucinating -- i.e. inserting fictitious events --the stacks or raw input streams compensating for the lack of information inthe absence of brightness changes. Our techniques are general can be adaptedto any structured representation to stack events and outperformstate-of-the-art fusion methods applied to event-based stereo. |


| Item |Content|
| --- |---|
|idx| 2408.04632v1 |
|title| Arctic-TILT. Business Document Understanding at Sub-Billion Scale |
|authors| Łukasz BorchmannMichał PietruszkaWojciech JaśkowskiDawid JurkiewiczPiotr HalamaPaweł JóziakŁukasz GarncarekPaweł LiskowskiKarolina SzyndlerAndrzej GretkowskiJulita OłtusekGabriela NowakowskaArtur ZawłockiŁukasz DuhrPaweł DydaMichał Turski
|links| http://arxiv.org/abs/2408.04632v1 |
|updated| 2024-08-08 17:59:46 UTC |
|summary| The vast portion of workloads employing LLMs involves answering questionsgrounded on PDF or scan content. We introduce the Arctic-TILT achievingaccuracy on par with models 1000times its size on these use cases. It can befine-tuned and deployed on a single 24GB GPU lowering operational costs whileprocessing Visually Rich Documents with up to 400k tokens. The modelestablishes state-of-the-art results on seven diverse Document Understandingbenchmarks as well as provides reliable confidence scores and quick inferencewhich are essential for processing files in large-scale or time-sensitiveenterprise environments. |


| Item |Content|
| --- |---|
|idx| 2408.04631v1 |
|title| Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics |
|authors| Ruining LiChuanxia ZhengChristian RupprechtAndrea Vedaldi
|links| http://arxiv.org/abs/2408.04631v1 |
|updated| 2024-08-08 17:59:38 UTC |
|summary| We present Puppet-Master an interactive video generative model that canserve as a motion prior for part-level dynamics. At test time given a singleimage and a sparse set of motion trajectories i.e. drags Puppet-Master cansynthesize a video depicting realistic part-level motion faithful to the givendrag interactions. This is achieved by fine-tuning a large-scale pre-trainedvideo diffusion model for which we propose a new conditioning architecture toinject the dragging control effectively. More importantly we introduce theall-to-first attention mechanism a drop-in replacement for the widely adoptedspatial attention modules which significantly improves generation quality byaddressing the appearance and background issues in existing models. Unlikeother motion-conditioned video generators that are trained on in-the-wildvideos and mostly move an entire object Puppet-Master is learned fromObjaverse-Animation-HQ a new dataset of curated part-level motion clips. Wepropose a strategy to automatically filter out sub-optimal animations andaugment the synthetic renderings with meaningful motion trajectories.Puppet-Master generalizes well to real images across various categories andoutperforms existing methods in a zero-shot manner on a real-world benchmark.See our project page for more results: vgg-puppetmaster.github.io. |


| Item |Content|
| --- |---|
|idx| 2408.04628v1 |
|title| LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP |
|authors| Danlu ChenFreda ShiAditi AgarwalJacobo MyerstonTaylor Berg-Kirkpatrick
|links| http://arxiv.org/abs/2408.04628v1 |
|updated| 2024-08-08 17:58:06 UTC |
|summary| Standard natural language processing NLP pipelines operate on symbolicrepresentations of language which typically consist of sequences of discretetokens. However creating an analogous representation for ancient logographicwriting systems is an extremely labor intensive process that requires expertknowledge. At present a large portion of logographic data persists in a purelyvisual form due to the absence of transcription -- this issue poses abottleneck for researchers seeking to apply NLP toolkits to study ancientlogographic languages: most of the relevant data are images of writing.  This paper investigates whether direct processing of visual representationsof language offers a potential solution. We introduce LogogramNLP the firstbenchmark enabling NLP analysis of ancient logographic languages featuringboth transcribed and visual datasets for four writing systems along withannotations for tasks like classification translation and parsing. Ourexperiments compare systems that employ recent visual and text encodingstrategies as backbones. The results demonstrate that visual representationsoutperform textual representations for some investigated tasks suggesting thatvisual processing pipelines may unlock a large amount of cultural heritage dataof logographic languages for NLP-based analyses. |


| Item |Content|
| --- |---|
|idx| 2408.04610v1 |
|title| Quantifying the Impact of Population Shift Across Age and Sex for Abdominal Organ Segmentation |
|authors| Kate ČevoraBen GlockerWenjia Bai
|links| http://arxiv.org/abs/2408.04610v1 |
|updated| 2024-08-08 17:28:32 UTC |
|summary| Deep learning-based medical image segmentation has seen tremendous progressover the last decade but there is still relatively little transfer intoclinical practice. One of the main barriers is the challenge of domaingeneralisation which requires segmentation models to maintain high performanceacross a wide distribution of image data. This challenge is amplified by themany factors that contribute to the diverse appearance of medical images suchas acquisition conditions and patient characteristics. The impact of shiftingpatient characteristics such as age and sex on segmentation performance remainsrelatively under-studied especially for abdominal organs despite that this iscrucial for ensuring the fairness of the segmentation model. We perform thefirst study to determine the impact of population shift with respect to age andsex on abdominal CT image segmentation by leveraging two large publicdatasets and introduce a novel metric to quantify the impact. We find thatpopulation shift is a challenge similar in magnitude to cross-dataset shift forabdominal organ segmentation and that the effect is asymmetric anddataset-dependent. We conclude that dataset diversity in terms of known patientcharacteristics is not necessarily equivalent to dataset diversity in terms ofimage features. This implies that simple population matching to ensure goodgeneralisation and fairness may be insufficient and we recommend that fairnessresearch should be directed towards better understanding and quantifyingmedical image dataset diversity in terms of performance-relevantcharacteristics such as organ morphology. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.04607v1 |
|title| Risk and cross validation in ridge regression with correlated samples |
|authors| Alexander AtanasovJacob A. Zavatone-VethCengiz Pehlevan
|links| http://arxiv.org/abs/2408.04607v1 |
|updated| 2024-08-08 17:27:29 UTC |
|summary| Recent years have seen substantial advances in our understanding ofhigh-dimensional ridge regression but existing theories assume that trainingexamples are independent. By leveraging recent techniques from random matrixtheory and free probability we provide sharp asymptotics for the in- andout-of-sample risks of ridge regression when the data points have arbitrarycorrelations. We demonstrate that in this setting the generalized crossvalidation estimator GCV fails to correctly predict the out-of-sample risk.However in the case where the noise residuals have the same correlations asthe data points one can modify the GCV to yield an efficiently-computableunbiased estimator that concentrates in the high-dimensional limit which wedub CorrGCV. We further extend our asymptotic analysis to the case where thetest point has nontrivial correlations with the training set a setting oftenencountered in time series forecasting. Assuming knowledge of the correlationstructure of the time series this again yields an extension of the GCVestimator and sharply characterizes the degree to which such test points yieldan overly optimistic prediction of long-time risk. We validate the predictionsof our theory across a variety of high dimensional data. |


| Item |Content|
| --- |---|
|idx| 2408.04595v1 |
|title| Inference with the Upper Confidence Bound Algorithm |
|authors| Koulik KhamaruCun-Hui Zhang
|links| http://arxiv.org/abs/2408.04595v1 |
|updated| 2024-08-08 17:11:36 UTC |
|summary| In this paper we discuss the asymptotic behavior of the Upper ConfidenceBound UCB algorithm in the context of multiarmed bandit problems and discussits implication in downstream inferential tasks. While inferential tasks becomechallenging when data is collected in a sequential manner we argue that thisproblem can be alleviated when the sequential algorithm at hand satisfiescertain stability property. This notion of stability is motivated from theseminal work of Lai and Wei 1982. Our first main result shows that such astability property is always satisfied for the UCB algorithm and as a resultthe sample means for each arm are asymptotically normal. Next we examine thestability properties of the UCB algorithm when the number of arms K isallowed to grow with the number of arm pulls T. We show that in such a casethe arms are stable when fraclog Klog T rightarrow 0 and the numberof near-optimal arms are large. |


| Item |Content|
| --- |---|
|idx| 2408.04569v1 |
|title| Activation thresholds and expressiveness of polynomial neural networks |
|authors| Bella FinkelJose Israel RodriguezChenxi WuThomas Yahl
|links| http://arxiv.org/abs/2408.04569v1 |
|updated| 2024-08-08 16:28:56 UTC |
|summary| Polynomial neural networks have been implemented in a range of applicationsand present an advantageous framework for theoretical machine learning. Apolynomial neural network of fixed architecture and activation degree gives analgebraic map from the networks weights to a set of polynomials. The image ofthis map is the space of functions representable by the network. Its Zariskiclosure is an affine variety known as a neurovariety. The dimension of apolynomial neural networks neurovariety provides a measure of itsexpressivity. In this work we introduce the notion of the activation thresholdof a network architecture which expresses when the dimension of a neurovarietyachieves its theoretical maximum. In addition we prove expressiveness resultsfor polynomial neural networks with equi-widtharchitectures. |


| Item |Content|
| --- |---|
|idx| 2408.04526v1 |
|title| Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs |
|authors| Kevin TanWei FanYuting Wei
|links| http://arxiv.org/abs/2408.04526v1 |
|updated| 2024-08-08 15:26:18 UTC |
|summary| Hybrid Reinforcement Learning RL where an agent learns from both anoffline dataset and online explorations in an unknown environment has garneredsignificant recent interest. A crucial question posed by Xie et al. 2022 iswhether hybrid RL can improve upon the existing lower bounds established inpurely offline and purely online RL without relying on the single-policyconcentrability assumption. While Li et al. 2023 provided an affirmativeanswer to this question in the tabular PAC RL case the question remainsunsettled for both the regret-minimizing RL case and the non-tabular case.  In this work building upon recent advancements in offline RL andreward-agnostic exploration we develop computationally efficient algorithmsfor both PAC and regret-minimizing RL with linear function approximationwithout single-policy concentrability. We demonstrate that these algorithmsachieve sharper error or regret bounds that are no worse than and can improveon the optimal sample complexity in offline RL the first algorithm for PACRL and online RL the second algorithm for regret-minimizing RL in linearMarkov decision processes MDPs regardless of the quality of the behaviorpolicy. To our knowledge this work establishes the tightest theoreticalguarantees currently available for hybrid RL in linear MDPs. |


| Item |Content|
| --- |---|
|idx| 2408.04391v1 |
|title| Robustness investigation of quality measures for the assessment of machine learning models |
|authors| Thomas MostLars GräningSebastian Wolff
|links| http://arxiv.org/abs/2408.04391v1 |
|updated| 2024-08-08 11:51:34 UTC |
|summary| In this paper the accuracy and robustness of quality measures for theassessment of machine learning models are investigated. The prediction qualityof a machine learning model is evaluated model-independent based on across-validation approach where the approximation error is estimated forunknown data. The presented measures quantify the amount of explained variationin the model prediction. The reliability of these measures is assessed by meansof several numerical examples where an additional data set for theverification of the estimated prediction error is available. Furthermore theconfidence bounds of the presented quality measures are estimated and localquality measures are derived from the prediction residuals obtained by thecross-validation approach. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.04619v1 |
|title| Transformer Explainer: Interactive Learning of Text-Generative Models |
|authors| Aeree ChoGrace C. KimAlexander KarpekovAlec HelblingZijie J. WangSeongmin LeeBenjamin HooverDuen Horng Chau
|links| http://arxiv.org/abs/2408.04619v1 |
|updated| 2024-08-08 17:49:07 UTC |
|summary| Transformers have revolutionized machine learning yet their inner workingsremain opaque to many. We present Transformer Explainer an interactivevisualization tool designed for non-experts to learn about Transformers throughthe GPT-2 model. Our tool helps users understand complex Transformer conceptsby integrating a model overview and enabling smooth transitions acrossabstraction levels of mathematical operations and model structures. It runs alive GPT-2 instance locally in the users browser empowering users toexperiment with their own input and observe in real-time how the internalcomponents and parameters of the Transformer work together to predict the nexttokens. Our tool requires no installation or special hardware broadening thepublics education access to modern generative AI techniques. Our open-sourcedtool is available at https://poloclub.github.io/transformer-explainer/. A videodemo is available at https://youtu.be/ECR4oAwocjs. |


| Item |Content|
| --- |---|
|idx| 2408.04574v1 |
|title| Integrating Annotations into the Design Process for Sonifications and Physicalizations |
|authors| Rhys Sorenson-GraffS. Sandra BaeJordan Wirfs-Brock
|links| http://arxiv.org/abs/2408.04574v1 |
|updated| 2024-08-08 16:36:14 UTC |
|summary| Annotations are a critical component of visualizations helping viewersinterpret the visual representation and highlighting critical data insights.Despite their significant role we lack an understanding of how annotations canbe incorporated into other data representations such as physicalizations andsonifications. Given the emergent nature of these representationssonifications and physicalizations lack formalized conventions e.g. designspace vocabulary that can introduce challenges for audiences to interpret theintended data encoding. To address this challenge this work focuses on howannotations can be more tightly integrated into the design process of creatingsonifications and physicalizations. In an exploratory study with 13 designerswe explore how visualization annotation techniques can be adapted to sonic andphysical modalities. Our work highlights how annotations for sonification andphysicalizations are inseparable from their data encodings. |


| Item |Content|
| --- |---|
|idx| 2408.04539v1 |
|title| ParetoTracker: Understanding Population Dynamics in Multi-objective Evolutionary Algorithms through Visual Analytics |
|authors| Zherui ZhangFan YangRan ChengYuxin Ma
|links| http://arxiv.org/abs/2408.04539v1 |
|updated| 2024-08-08 15:46:11 UTC |
|summary| Multi-objective evolutionary algorithms MOEAs have emerged as powerfultools for solving complex optimization problems characterized by multipleoften conflicting objectives. While advancements have been made incomputational efficiency as well as diversity and convergence of solutions acritical challenge persists: the internal evolutionary mechanisms are opaque tohuman users. Drawing upon the successes of explainable AI in explaining complexalgorithms and models we argue that the need to understand the underlyingevolutionary operators and population dynamics within MOEAs aligns well with avisual analytics paradigm. This paper introduces ParetoTracker a visualanalytics framework designed to support the comprehension and inspection ofpopulation dynamics in the evolutionary processes of MOEAs. Informed bypreliminary literature review and expert interviews the framework establishesa multi-level analysis scheme which caters to user engagement and explorationranging from examining overall trends in performance metrics to conductingfine-grained inspections of evolutionary operations. In contrast toconventional practices that require manual plotting of solutions for eachgeneration ParetoTracker facilitates the examination of temporal trends anddynamics across consecutive generations in an integrated visual interface. Theeffectiveness of the framework is demonstrated through case studies and expertinterviews focused on widely adopted benchmark optimization problems. |


| Item |Content|
| --- |---|
|idx| 2408.04506v1 |
|title| Who ruins the game?: unveiling cheating players in the "Battlefield" game |
|authors| Dong Young KimHuy Kang Kim
|links| http://arxiv.org/abs/2408.04506v1 |
|updated| 2024-08-08 15:04:23 UTC |
|summary| The Battlefield online game is well-known for its large-scale multiplayercapabilities and unique gaming features including various vehicle controls.However these features make the game a major target for cheatingsignificantly detracting from the gaming experience. This study analyzes userbehavior in cheating play in the popular online game the Battlefield usingstatistical methods. We aim to provide comprehensive insights into cheatingplayers through an extensive analysis of over 44000 reported cheatingincidents collected via the Game-tools API. Our methodology includes detailedstatistical analyses such as calculating basic statistics of key variablescorrelation analysis and visualizations using histograms box plots andscatter plots. Our findings emphasize the importance of adaptive data-drivenapproaches to prevent cheating plays in online games. |


| Item |Content|
| --- |---|
|idx| 2408.04500v1 |
|title| "I Am Human, Just Like You": What Intersectional, Neurodivergent Lived Experiences Bring to Accessibility Research |
|authors| Lindy Le
|links| http://dx.doi.org/10.1145/3663548.3675651 |
|updated| 2024-08-08 14:55:13 UTC |
|summary| The increasing prevalence of neurodivergence has led society to give greaterrecognition to the importance of neurodiversity. Yet societal perceptions ofneurodivergence continue to be predominantly negative. Drawing on CriticalDisability Studies accessibility researchers have demonstrated howneuronormative assumptions dominate HCI. Despite their guidance neurodivergentand disabled individuals are still marginalized in technology research. Inparticular intersectional identities remain largely absent from HCIneurodivergence research. In this paper I share my perspective as an outsiderof the academic research community: I use critical autoethnography to analyzemy experiences of coming to understand accept and value my neurodivergencewithin systems of power privilege and oppression. Using Data Feminism as anaccessible and practical guide to intersectionality I derive three tenets forreconceptualizing neurodivergence to be more inclusive of intersectionalexperiences: 1 neurodivergence is a functional difference not a deficit 2neurodivergent disability is a moment of friction not a static label and 3neurodivergence accessibility is a collaborative practice not a one-sidedsolution. Then I discuss the tenets in the context of existing HCI researchapplying the same intersectional lens. Finally I offer three suggestions forhow accessibility research can apply these tenets in future work to bridge thegap between accessibility theory and practice in HCI neurodivergence research |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.04549v1 |
|title| Learning Fair Cooperation in Mixed-Motive Games with Indirect Reciprocity |
|authors| Martin SmitFernando P. Santos
|links| http://dx.doi.org/10.24963/ijcai.2024/25 |
|updated| 2024-08-08 15:57:15 UTC |
|summary| Altruistic cooperation is costly yet socially desirable. As a result agentsstruggle to learn cooperative policies through independent reinforcementlearning RL. Indirect reciprocity where agents consider their interactionpartners reputation has been shown to stabilise cooperation in homogeneousidealised populations. However more realistic settings are comprised ofheterogeneous agents with different characteristics and group-based socialidentities. We study cooperation when agents are stratified into two suchgroups and allow reputation updates and actions to depend on groupinformation. We consider two modelling approaches: evolutionary game theorywhere we comprehensively search for social norms i.e. rules to assignreputations leading to cooperation and fairness and RL where we consider howthe stochastic dynamics of policy learning affects the analytically identifiedequilibria. We observe that a defecting majority leads the minority group todefect but not the inverse. Moreover changing the norms that judge in andout-group interactions can steer a system towards either fair or unfaircooperation. This is made clearer when moving beyond equilibrium analysis toindependent RL agents where convergence to fair cooperation occurs with anarrower set of norms. Our results highlight that in heterogeneous populationswith reputations carefully defining interaction norms is fundamental to tackleboth dilemmas of cooperation and of fairness. |


| Item |Content|
| --- |---|
|idx| 2408.04514v1 |
|title| Emergence in Multi-Agent Systems: A Safety Perspective |
|authors| Philipp AltmannJulian SchönbergerSteffen IlliumMaximilian ZornFabian RitzTom HaiderSimon BurtonThomas Gabor
|links| http://arxiv.org/abs/2408.04514v1 |
|updated| 2024-08-08 15:15:28 UTC |
|summary| Emergent effects can arise in multi-agent systems MAS where execution isdecentralized and reliant on local information. These effects may range fromminor deviations in behavior to catastrophic system failures. To formallydefine these effects we identify misalignments between the global inherentspecification the true specification and its local approximation such as theconfiguration of different reward components or observations. Usingestablished safety terminology we develop a framework to understand theseemergent effects. To showcase the resulting implications we use two broadlyconfigurable exemplary gridworld scenarios where insufficient specificationleads to unintended behavior deviations when derived independently. Recognizingthat a global adaptation might not always be feasible we propose adjusting theunderlying parameterizations to mitigate these issues thereby improving thesystems alignment and reducing the risk of emergent failures. |


| Item |Content|
| --- |---|
|idx| 2408.04295v1 |
|title| Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization |
|authors| Aditya KapoorBenjamin FreedHowie ChosetJeff Schneider
|links| http://arxiv.org/abs/2408.04295v1 |
|updated| 2024-08-08 08:18:05 UTC |
|summary| Multi-agent proximal policy optimization MAPPO has recently demonstratedstate-of-the-art performance on challenging multi-agent reinforcement learningtasks. However MAPPO still struggles with the credit assignment problemwherein the sheer difficulty in ascribing credit to individual agents actionsscales poorly with team size. In this paper we propose a multi-agentreinforcement learning algorithm that adapts recent developments in creditassignment to improve upon MAPPO. Our approach leverages partial rewarddecoupling PRD which uses a learned attention mechanism to estimate which ofa particular agents teammates are relevant to its learning updates. We usethis estimate to dynamically decompose large groups of agents into smallermore manageable subgroups. We empirically demonstrate that our approachPRD-MAPPO decouples agents from teammates that do not influence their expectedfuture reward thereby streamlining credit assignment. We additionally showthat PRD-MAPPO yields significantly higher data efficiency and asymptoticperformance compared to both MAPPO and other state-of-the-art methods acrossseveral multi-agent tasks including StarCraft II. Finally we propose aversion of PRD-MAPPO that is applicable to textitshared reward settingswhere PRD was previously not applicable and empirically show that this alsoleads to performance improvements over MAPPO. |


| Item |Content|
| --- |---|
|idx| 2408.03692v1 |
|title| Asynchronous Credit Assignment Framework for Multi-Agent Reinforcement Learning |
|authors| Yongheng LiangHejun WuHaitao WangHao Cai
|links| http://arxiv.org/abs/2408.03692v1 |
|updated| 2024-08-07 11:13:26 UTC |
|summary| Credit assignment is a core problem that distinguishes agents marginalcontributions for optimizing cooperative strategies in multi-agentreinforcement learning MARL. Current credit assignment methods usually assumesynchronous decision-making among agents. However a prerequisite for manyrealistic cooperative tasks is asynchronous decision-making by agents withoutwaiting for others to avoid disastrous consequences. To address this issue wepropose an asynchronous credit assignment framework with a problem model calledADEX-POMDP and a multiplicative value decomposition MVD algorithm. ADEX-POMDPis an asynchronous problem model with extra virtual agents for a decentralizedpartially observable markov decision process. We prove that ADEX-POMDPpreserves both the task equilibrium and the algorithm convergence. MVD utilizesmultiplicative interaction to efficiently capture the interactions ofasynchronous decisions and we theoretically demonstrate its advantages inhandling asynchronous tasks. Experimental results show that on two asynchronousdecision-making benchmarks Overcooked and POAC MVD not only consistentlyoutperforms state-of-the-art MARL methods but also provides theinterpretability for asynchronous cooperation. |


| Item |Content|
| --- |---|
|idx| 2408.03405v1 |
|title| Combining Diverse Information for Coordinated Action: Stochastic Bandit Algorithms for Heterogeneous Agents |
|authors| Lucia GordonEsther RolfMilind Tambe
|links| http://arxiv.org/abs/2408.03405v1 |
|updated| 2024-08-06 18:56:29 UTC |
|summary| Stochastic multi-agent multi-armed bandits typically assume that the rewardsfrom each arm follow a fixed distribution regardless of which agent pulls thearm. However in many real-world settings rewards can depend on thesensitivity of each agent to their environment. In medical screening diseasedetection rates can vary by test type in preference matching rewards candepend on user preferences and in environmental sensing observation qualitycan vary across sensors. Since past work does not specify how to allocateagents of heterogeneous but known sensitivity of these types in a stochasticbandit setting we introduce a UCB-style algorithm Min-Width which aggregatesinformation from diverse agents. In doing so we address the joint challengesof i aggregating the rewards which follow different distributions for eachagent-arm pair and ii coordinating the assignments of agents to arms.Min-Width facilitates efficient collaboration among heterogeneous agentsexploiting the known structure in the agents reward functions to weight theirrewards accordingly. We analyze the regret of Min-Width and conductpseudo-synthetic and fully synthetic experiments to study the performance ofdifferent levels of information sharing. Our results confirm that the gains tomodeling agent heterogeneity tend to be greater when the sensitivities are morevaried across agents while combining more information does not always improveperformance. |


