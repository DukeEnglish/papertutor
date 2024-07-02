# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.20098v1 |
|title| Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs |
|authors| Sukmin YunHaokun LinRusiru ThusharaMohammad Qazim BhatYongxin WangZutao JiangMingkai DengJinhong WangTianhua TaoJunbo LiHaonan LiPreslav NakovTimothy BaldwinZhengzhong LiuEric P. XingXiaodan LiangZhiqiang Shen
|links| http://arxiv.org/abs/2406.20098v1 |
|updated| 2024-06-28 17:59:46 UTC |
|summary| Multimodal large language models MLLMs have shown impressive success acrossmodalities such as image video and audio in a variety of understanding andgeneration tasks. However current MLLMs are surprisingly poor at understandingwebpage screenshots and generating their corresponding HTML code. To addressthis problem we propose Web2Code a benchmark consisting of a new large-scalewebpage-to-code dataset for instruction tuning and an evaluation framework forthe webpage understanding and HTML code translation abilities of MLLMs. Fordataset construction we leverage pretrained LLMs to enhance existingwebpage-to-code datasets as well as generate a diverse pool of new webpagesrendered into images. Specifically the inputs are webpage images andinstructions while the responses are the webpages HTML code. We furtherinclude diverse natural language QA pairs about the webpage content in theresponses to enable a more comprehensive understanding of the web content. Toevaluate model performance in these tasks we develop an evaluation frameworkfor testing MLLMs abilities in webpage understanding and web-to-codegeneration. Extensive experiments show that our proposed dataset is beneficialnot only to our proposed tasks but also in the general visual domain whileprevious datasets result in worse performance. We hope our work will contributeto the development of general MLLMs suitable for web-based content generationand task automation. Our data and code will be available athttps://github.com/MBZUAI-LLM/web2code. |


| Item |Content|
| --- |---|
|idx| 2406.20095v1 |
|title| LLaRA: Supercharging Robot Learning Data for Vision-Language Policy |
|authors| Xiang LiCristina MataJongwoo ParkKumara KahatapitiyaYoo Sung JangJinghuan ShangKanchana RanasingheRyan BurgertMu CaiYong Jae LeeMichael S. Ryoo
|links| http://arxiv.org/abs/2406.20095v1 |
|updated| 2024-06-28 17:59:12 UTC |
|summary| Large Language Models LLMs equipped with extensive world knowledge andstrong reasoning skills can tackle diverse tasks across domains often byposing them as conversation-style instruction-response pairs. In this paper wepropose LLaRA: Large Language and Robotics Assistant a framework whichformulates robot action policy as conversations and provides improvedresponses when trained with auxiliary data that complements policy learning.LLMs with visual inputs i.e. Vision Language Models VLMs have the capacityto process state information as visual-textual prompts and generate optimalpolicy decisions in text. To train such action policy VLMs we first introducean automated pipeline to generate diverse high-quality robotics instructiondata from existing behavior cloning data. A VLM finetuned with the resultingcollection of datasets based on a conversation-style formulation tailored forrobotics tasks can generate meaningful robot action policy decisions. Ourexperiments across multiple simulated and real-world environments demonstratethe state-of-the-art performance of the proposed LLaRA framework. The codedatasets and pretrained models are available athttps://github.com/LostXine/LLaRA. |


| Item |Content|
| --- |---|
|idx| 2406.20094v1 |
|title| Scaling Synthetic Data Creation with 1,000,000,000 Personas |
|authors| Xin ChanXiaoyang WangDian YuHaitao MiDong Yu
|links| http://arxiv.org/abs/2406.20094v1 |
|updated| 2024-06-28 17:59:01 UTC |
|summary| We propose a novel persona-driven data synthesis methodology that leveragesvarious perspectives within a large language model LLM to create diversesynthetic data. To fully exploit this methodology at scale we introducePersona Hub -- a collection of 1 billion diverse personas automatically curatedfrom web data. These 1 billion personas 13 of the worlds total populationacting as distributed carriers of world knowledge can tap into almost everyperspective encapsulated within the LLM thereby facilitating the creation ofdiverse synthetic data at scale for various scenarios. By showcasing PersonaHubs use cases in synthesizing high-quality mathematical and logical reasoningproblems instructions i.e. user prompts knowledge-rich texts game NPCsand tools functions at scale we demonstrate persona-driven data synthesis isversatile scalable flexible and easy to use potentially driving a paradigmshift in synthetic data creation and applications in practice which may have aprofound impact on LLM research and development. |


| Item |Content|
| --- |---|
|idx| 2406.20087v1 |
|title| ProgressGym: Alignment with a Millennium of Moral Progress |
|authors| Tianyi QiuYang ZhangXuchuan HuangJasmine Xinze LiJiaming JiYaodong Yang
|links| http://arxiv.org/abs/2406.20087v1 |
|updated| 2024-06-28 17:55:24 UTC |
|summary| Frontier AI systems including large language models LLMs hold increasinginfluence over the epistemology of human users. Such influence can reinforceprevailing societal values potentially contributing to the lock-in ofmisguided moral beliefs and consequently the perpetuation of problematicmoral practices on a broad scale. We introduce progress alignment as atechnical solution to mitigate this imminent risk. Progress alignmentalgorithms learn to emulate the mechanics of human moral progress therebyaddressing the susceptibility of existing alignment methods to contemporarymoral blindspots. To empower research in progress alignment we introduceProgressGym an experimental framework allowing the learning of moral progressmechanics from history in order to facilitate future progress in real-worldmoral decisions. Leveraging 9 centuries of historical text and 18 historicalLLMs ProgressGym enables codification of real-world progress alignmentchallenges into concrete benchmarks. Specifically we introduce three corechallenges: tracking evolving values PG-Follow preemptively anticipatingmoral progress PG-Predict and regulating the feedback loop between human andAI value shifts PG-Coevolve. Alignment methods without a temporal dimensionare inapplicable to these tasks. In response we present lifelong andextrapolative algorithms as baseline methods of progress alignment and buildan open leaderboard soliciting novel algorithms and challenges. The frameworkand the leaderboard are available athttps://github.com/PKU-Alignment/ProgressGym andhttps://huggingface.co/spaces/PKU-Alignment/ProgressGym-LeaderBoardrespectively. |


| Item |Content|
| --- |---|
|idx| 2406.20086v1 |
|title| Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs |
|authors| Sheridan FeuchtDavid AtkinsonByron WallaceDavid Bau
|links| http://arxiv.org/abs/2406.20086v1 |
|updated| 2024-06-28 17:54:47 UTC |
|summary| LLMs process text as sequences of tokens that roughly correspond to wordswhere less common words are represented by multiple tokens. However individualtokens are often semantically unrelated to the meanings of the words/conceptsthey comprise. For example Llama-2-7bs tokenizer splits the wordnortheastern into the tokens _n ort he astern none of whichcorrespond to semantically meaningful units like north or east. Similarlythe overall meanings of named entities like Neil Young and multi-wordexpressions like break a leg cannot be directly inferred from theirconstituent tokens. Mechanistically how do LLMs convert such arbitrary groupsof tokens into useful higher-level representations In this work we find thatlast token representations of named entities and multi-token words exhibit apronounced erasure effect where information about previous and currenttokens is rapidly forgotten in early layers. Using this observation we proposea method to read out the implicit vocabulary of an autoregressive LLM byexamining differences in token representations across layers and presentresults of this method for Llama-2-7b and Llama-3-8B. To our knowledge this isthe first attempt to probe the implicit vocabulary of an LLM. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.20098v1 |
|title| Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs |
|authors| Sukmin YunHaokun LinRusiru ThusharaMohammad Qazim BhatYongxin WangZutao JiangMingkai DengJinhong WangTianhua TaoJunbo LiHaonan LiPreslav NakovTimothy BaldwinZhengzhong LiuEric P. XingXiaodan LiangZhiqiang Shen
|links| http://arxiv.org/abs/2406.20098v1 |
|updated| 2024-06-28 17:59:46 UTC |
|summary| Multimodal large language models MLLMs have shown impressive success acrossmodalities such as image video and audio in a variety of understanding andgeneration tasks. However current MLLMs are surprisingly poor at understandingwebpage screenshots and generating their corresponding HTML code. To addressthis problem we propose Web2Code a benchmark consisting of a new large-scalewebpage-to-code dataset for instruction tuning and an evaluation framework forthe webpage understanding and HTML code translation abilities of MLLMs. Fordataset construction we leverage pretrained LLMs to enhance existingwebpage-to-code datasets as well as generate a diverse pool of new webpagesrendered into images. Specifically the inputs are webpage images andinstructions while the responses are the webpages HTML code. We furtherinclude diverse natural language QA pairs about the webpage content in theresponses to enable a more comprehensive understanding of the web content. Toevaluate model performance in these tasks we develop an evaluation frameworkfor testing MLLMs abilities in webpage understanding and web-to-codegeneration. Extensive experiments show that our proposed dataset is beneficialnot only to our proposed tasks but also in the general visual domain whileprevious datasets result in worse performance. We hope our work will contributeto the development of general MLLMs suitable for web-based content generationand task automation. Our data and code will be available athttps://github.com/MBZUAI-LLM/web2code. |


| Item |Content|
| --- |---|
|idx| 2406.20095v1 |
|title| LLaRA: Supercharging Robot Learning Data for Vision-Language Policy |
|authors| Xiang LiCristina MataJongwoo ParkKumara KahatapitiyaYoo Sung JangJinghuan ShangKanchana RanasingheRyan BurgertMu CaiYong Jae LeeMichael S. Ryoo
|links| http://arxiv.org/abs/2406.20095v1 |
|updated| 2024-06-28 17:59:12 UTC |
|summary| Large Language Models LLMs equipped with extensive world knowledge andstrong reasoning skills can tackle diverse tasks across domains often byposing them as conversation-style instruction-response pairs. In this paper wepropose LLaRA: Large Language and Robotics Assistant a framework whichformulates robot action policy as conversations and provides improvedresponses when trained with auxiliary data that complements policy learning.LLMs with visual inputs i.e. Vision Language Models VLMs have the capacityto process state information as visual-textual prompts and generate optimalpolicy decisions in text. To train such action policy VLMs we first introducean automated pipeline to generate diverse high-quality robotics instructiondata from existing behavior cloning data. A VLM finetuned with the resultingcollection of datasets based on a conversation-style formulation tailored forrobotics tasks can generate meaningful robot action policy decisions. Ourexperiments across multiple simulated and real-world environments demonstratethe state-of-the-art performance of the proposed LLaRA framework. The codedatasets and pretrained models are available athttps://github.com/LostXine/LLaRA. |


| Item |Content|
| --- |---|
|idx| 2406.20087v1 |
|title| ProgressGym: Alignment with a Millennium of Moral Progress |
|authors| Tianyi QiuYang ZhangXuchuan HuangJasmine Xinze LiJiaming JiYaodong Yang
|links| http://arxiv.org/abs/2406.20087v1 |
|updated| 2024-06-28 17:55:24 UTC |
|summary| Frontier AI systems including large language models LLMs hold increasinginfluence over the epistemology of human users. Such influence can reinforceprevailing societal values potentially contributing to the lock-in ofmisguided moral beliefs and consequently the perpetuation of problematicmoral practices on a broad scale. We introduce progress alignment as atechnical solution to mitigate this imminent risk. Progress alignmentalgorithms learn to emulate the mechanics of human moral progress therebyaddressing the susceptibility of existing alignment methods to contemporarymoral blindspots. To empower research in progress alignment we introduceProgressGym an experimental framework allowing the learning of moral progressmechanics from history in order to facilitate future progress in real-worldmoral decisions. Leveraging 9 centuries of historical text and 18 historicalLLMs ProgressGym enables codification of real-world progress alignmentchallenges into concrete benchmarks. Specifically we introduce three corechallenges: tracking evolving values PG-Follow preemptively anticipatingmoral progress PG-Predict and regulating the feedback loop between human andAI value shifts PG-Coevolve. Alignment methods without a temporal dimensionare inapplicable to these tasks. In response we present lifelong andextrapolative algorithms as baseline methods of progress alignment and buildan open leaderboard soliciting novel algorithms and challenges. The frameworkand the leaderboard are available athttps://github.com/PKU-Alignment/ProgressGym andhttps://huggingface.co/spaces/PKU-Alignment/ProgressGym-LeaderBoardrespectively. |


| Item |Content|
| --- |---|
|idx| 2406.20080v1 |
|title| AI for Extreme Event Modeling and Understanding: Methodologies and Challenges |
|authors| Gustau Camps-VallsMiguel-Ángel Fernández-TorresKai-Hendrik CohrsAdrian HöhlAndrea CastellettiAytac PacalClaire RobinFrancesco MartinuzziIoannis PapoutsisIoannis PrapasJorge Pérez-AracilKatja WeigelMaria Gonzalez-CalabuigMarkus ReichsteinMartin RabelMatteo GiulianiMiguel MahechaOana-Iuliana PopescuOscar J. Pellicer-ValeroSaid OualaSancho Salcedo-SanzSebastian SippelSpyros KondylatosTamara HappéTristan Williams
|links| http://arxiv.org/abs/2406.20080v1 |
|updated| 2024-06-28 17:45:25 UTC |
|summary| In recent years artificial intelligence AI has deeply impacted variousfields including Earth system sciences. Here AI improved weather forecastingmodel emulation parameter estimation and the prediction of extreme events.However the latter comes with specific challenges such as developing accuratepredictors from noisy heterogeneous and limited annotated data. This paperreviews how AI is being used to analyze extreme events like floods droughtswildfires and heatwaves highlighting the importance of creating accuratetransparent and reliable AI models. We discuss the hurdles of dealing withlimited data integrating information in real-time deploying models andmaking them understandable all crucial for gaining the trust of stakeholdersand meeting regulatory needs. We provide an overview of how AI can helpidentify and explain extreme events more effectively improving disasterresponse and communication. We emphasize the need for collaboration acrossdifferent fields to create AI solutions that are practical understandable andtrustworthy for analyzing and predicting extreme events. Such collaborativeefforts aim to enhance disaster readiness and disaster risk reduction. |


| Item |Content|
| --- |---|
|idx| 2406.20079v1 |
|title| Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification |
|authors| Anisha GunjalGreg Durrett
|links| http://arxiv.org/abs/2406.20079v1 |
|updated| 2024-06-28 17:43:48 UTC |
|summary| Automatic factuality verification of large language model LLM generationsis becoming more and more widely used to combat hallucinations. A major pointof tension in the literature is the granularity of this fact-checking: largerchunks of text are hard to fact-check but more atomic facts like propositionsmay lack context to interpret correctly. In this work we assess the role ofcontext in these atomic facts. We argue that fully atomic facts are not theright representation and define two criteria for molecular facts:decontextuality or how well they can stand alone and minimality or howlittle extra information is added to achieve decontexuality. We quantify theimpact of decontextualization on minimality then present a baselinemethodology for generating molecular facts automatically aiming to add theright amount of information. We compare against various methods ofdecontextualization and find that molecular facts balance minimality with factverification accuracy in ambiguous settings. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.20095v1 |
|title| LLaRA: Supercharging Robot Learning Data for Vision-Language Policy |
|authors| Xiang LiCristina MataJongwoo ParkKumara KahatapitiyaYoo Sung JangJinghuan ShangKanchana RanasingheRyan BurgertMu CaiYong Jae LeeMichael S. Ryoo
|links| http://arxiv.org/abs/2406.20095v1 |
|updated| 2024-06-28 17:59:12 UTC |
|summary| Large Language Models LLMs equipped with extensive world knowledge andstrong reasoning skills can tackle diverse tasks across domains often byposing them as conversation-style instruction-response pairs. In this paper wepropose LLaRA: Large Language and Robotics Assistant a framework whichformulates robot action policy as conversations and provides improvedresponses when trained with auxiliary data that complements policy learning.LLMs with visual inputs i.e. Vision Language Models VLMs have the capacityto process state information as visual-textual prompts and generate optimalpolicy decisions in text. To train such action policy VLMs we first introducean automated pipeline to generate diverse high-quality robotics instructiondata from existing behavior cloning data. A VLM finetuned with the resultingcollection of datasets based on a conversation-style formulation tailored forrobotics tasks can generate meaningful robot action policy decisions. Ourexperiments across multiple simulated and real-world environments demonstratethe state-of-the-art performance of the proposed LLaRA framework. The codedatasets and pretrained models are available athttps://github.com/LostXine/LLaRA. |


| Item |Content|
| --- |---|
|idx| 2406.20094v1 |
|title| Scaling Synthetic Data Creation with 1,000,000,000 Personas |
|authors| Xin ChanXiaoyang WangDian YuHaitao MiDong Yu
|links| http://arxiv.org/abs/2406.20094v1 |
|updated| 2024-06-28 17:59:01 UTC |
|summary| We propose a novel persona-driven data synthesis methodology that leveragesvarious perspectives within a large language model LLM to create diversesynthetic data. To fully exploit this methodology at scale we introducePersona Hub -- a collection of 1 billion diverse personas automatically curatedfrom web data. These 1 billion personas 13 of the worlds total populationacting as distributed carriers of world knowledge can tap into almost everyperspective encapsulated within the LLM thereby facilitating the creation ofdiverse synthetic data at scale for various scenarios. By showcasing PersonaHubs use cases in synthesizing high-quality mathematical and logical reasoningproblems instructions i.e. user prompts knowledge-rich texts game NPCsand tools functions at scale we demonstrate persona-driven data synthesis isversatile scalable flexible and easy to use potentially driving a paradigmshift in synthetic data creation and applications in practice which may have aprofound impact on LLM research and development. |


| Item |Content|
| --- |---|
|idx| 2406.20087v1 |
|title| ProgressGym: Alignment with a Millennium of Moral Progress |
|authors| Tianyi QiuYang ZhangXuchuan HuangJasmine Xinze LiJiaming JiYaodong Yang
|links| http://arxiv.org/abs/2406.20087v1 |
|updated| 2024-06-28 17:55:24 UTC |
|summary| Frontier AI systems including large language models LLMs hold increasinginfluence over the epistemology of human users. Such influence can reinforceprevailing societal values potentially contributing to the lock-in ofmisguided moral beliefs and consequently the perpetuation of problematicmoral practices on a broad scale. We introduce progress alignment as atechnical solution to mitigate this imminent risk. Progress alignmentalgorithms learn to emulate the mechanics of human moral progress therebyaddressing the susceptibility of existing alignment methods to contemporarymoral blindspots. To empower research in progress alignment we introduceProgressGym an experimental framework allowing the learning of moral progressmechanics from history in order to facilitate future progress in real-worldmoral decisions. Leveraging 9 centuries of historical text and 18 historicalLLMs ProgressGym enables codification of real-world progress alignmentchallenges into concrete benchmarks. Specifically we introduce three corechallenges: tracking evolving values PG-Follow preemptively anticipatingmoral progress PG-Predict and regulating the feedback loop between human andAI value shifts PG-Coevolve. Alignment methods without a temporal dimensionare inapplicable to these tasks. In response we present lifelong andextrapolative algorithms as baseline methods of progress alignment and buildan open leaderboard soliciting novel algorithms and challenges. The frameworkand the leaderboard are available athttps://github.com/PKU-Alignment/ProgressGym andhttps://huggingface.co/spaces/PKU-Alignment/ProgressGym-LeaderBoardrespectively. |


| Item |Content|
| --- |---|
|idx| 2406.20086v1 |
|title| Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs |
|authors| Sheridan FeuchtDavid AtkinsonByron WallaceDavid Bau
|links| http://arxiv.org/abs/2406.20086v1 |
|updated| 2024-06-28 17:54:47 UTC |
|summary| LLMs process text as sequences of tokens that roughly correspond to wordswhere less common words are represented by multiple tokens. However individualtokens are often semantically unrelated to the meanings of the words/conceptsthey comprise. For example Llama-2-7bs tokenizer splits the wordnortheastern into the tokens _n ort he astern none of whichcorrespond to semantically meaningful units like north or east. Similarlythe overall meanings of named entities like Neil Young and multi-wordexpressions like break a leg cannot be directly inferred from theirconstituent tokens. Mechanistically how do LLMs convert such arbitrary groupsof tokens into useful higher-level representations In this work we find thatlast token representations of named entities and multi-token words exhibit apronounced erasure effect where information about previous and currenttokens is rapidly forgotten in early layers. Using this observation we proposea method to read out the implicit vocabulary of an autoregressive LLM byexamining differences in token representations across layers and presentresults of this method for Llama-2-7b and Llama-3-8B. To our knowledge this isthe first attempt to probe the implicit vocabulary of an LLM. |


| Item |Content|
| --- |---|
|idx| 2406.20081v1 |
|title| Segment Anything without Supervision |
|authors| XuDong WangJingfeng YangTrevor Darrell
|links| http://arxiv.org/abs/2406.20081v1 |
|updated| 2024-06-28 17:47:32 UTC |
|summary| The Segmentation Anything Model SAM requires labor-intensive data labeling.We present Unsupervised SAM UnSAM for promptable and automatic whole-imagesegmentation that does not require human annotations. UnSAM utilizes adivide-and-conquer strategy to discover the hierarchical structure of visualscenes. We first leverage top-down clustering methods to partition an unlabeledimage into instance/semantic level segments. For all pixels within a segment abottom-up clustering method is employed to iteratively merge them into largergroups thereby forming a hierarchical structure. These unsupervisedmulti-granular masks are then utilized to supervise model training. Evaluatedacross seven popular datasets UnSAM achieves competitive results with thesupervised counterpart SAM and surpasses the previous state-of-the-art inunsupervised segmentation by 11 in terms of AR. Moreover we show thatsupervised SAM can also benefit from our self-supervised labels. By integratingour unsupervised pseudo masks into SA-1Bs ground-truth masks and trainingUnSAM with only 1 of SA-1B a lightly semi-supervised UnSAM can often segmententities overlooked by supervised SAM exceeding SAMs AR by over 6.7 and APby 3.9 on SA-1B. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.20099v1 |
|title| Odd-One-Out: Anomaly Detection by Comparing with Neighbors |
|authors| Ankan BhuniaChangjian LiHakan Bilen
|links| http://arxiv.org/abs/2406.20099v1 |
|updated| 2024-06-28 17:59:51 UTC |
|summary| This paper introduces a novel anomaly detection AD problem that focuses onidentifying odd-looking objects relative to the other instances within ascene. Unlike the traditional AD benchmarks in our setting anomalies in thiscontext are scene-specific defined by the regular instances that make up themajority. Since object instances are often partly visible from a singleviewpoint our setting provides multiple views of each scene as input. Toprovide a testbed for future research in this task we introduce twobenchmarks ToysAD-8K and PartsAD-15K. We propose a novel method that generates3D object-centric representations for each instance and detects the anomalousones through a cross-examination between the instances. We rigorously analyzeour method quantitatively and qualitatively in the presented benchmarks. |


| Item |Content|
| --- |---|
|idx| 2406.20098v1 |
|title| Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs |
|authors| Sukmin YunHaokun LinRusiru ThusharaMohammad Qazim BhatYongxin WangZutao JiangMingkai DengJinhong WangTianhua TaoJunbo LiHaonan LiPreslav NakovTimothy BaldwinZhengzhong LiuEric P. XingXiaodan LiangZhiqiang Shen
|links| http://arxiv.org/abs/2406.20098v1 |
|updated| 2024-06-28 17:59:46 UTC |
|summary| Multimodal large language models MLLMs have shown impressive success acrossmodalities such as image video and audio in a variety of understanding andgeneration tasks. However current MLLMs are surprisingly poor at understandingwebpage screenshots and generating their corresponding HTML code. To addressthis problem we propose Web2Code a benchmark consisting of a new large-scalewebpage-to-code dataset for instruction tuning and an evaluation framework forthe webpage understanding and HTML code translation abilities of MLLMs. Fordataset construction we leverage pretrained LLMs to enhance existingwebpage-to-code datasets as well as generate a diverse pool of new webpagesrendered into images. Specifically the inputs are webpage images andinstructions while the responses are the webpages HTML code. We furtherinclude diverse natural language QA pairs about the webpage content in theresponses to enable a more comprehensive understanding of the web content. Toevaluate model performance in these tasks we develop an evaluation frameworkfor testing MLLMs abilities in webpage understanding and web-to-codegeneration. Extensive experiments show that our proposed dataset is beneficialnot only to our proposed tasks but also in the general visual domain whileprevious datasets result in worse performance. We hope our work will contributeto the development of general MLLMs suitable for web-based content generationand task automation. Our data and code will be available athttps://github.com/MBZUAI-LLM/web2code. |


| Item |Content|
| --- |---|
|idx| 2406.20095v1 |
|title| LLaRA: Supercharging Robot Learning Data for Vision-Language Policy |
|authors| Xiang LiCristina MataJongwoo ParkKumara KahatapitiyaYoo Sung JangJinghuan ShangKanchana RanasingheRyan BurgertMu CaiYong Jae LeeMichael S. Ryoo
|links| http://arxiv.org/abs/2406.20095v1 |
|updated| 2024-06-28 17:59:12 UTC |
|summary| Large Language Models LLMs equipped with extensive world knowledge andstrong reasoning skills can tackle diverse tasks across domains often byposing them as conversation-style instruction-response pairs. In this paper wepropose LLaRA: Large Language and Robotics Assistant a framework whichformulates robot action policy as conversations and provides improvedresponses when trained with auxiliary data that complements policy learning.LLMs with visual inputs i.e. Vision Language Models VLMs have the capacityto process state information as visual-textual prompts and generate optimalpolicy decisions in text. To train such action policy VLMs we first introducean automated pipeline to generate diverse high-quality robotics instructiondata from existing behavior cloning data. A VLM finetuned with the resultingcollection of datasets based on a conversation-style formulation tailored forrobotics tasks can generate meaningful robot action policy decisions. Ourexperiments across multiple simulated and real-world environments demonstratethe state-of-the-art performance of the proposed LLaRA framework. The codedatasets and pretrained models are available athttps://github.com/LostXine/LLaRA. |


| Item |Content|
| --- |---|
|idx| 2406.20092v1 |
|title| LLaVolta: Efficient Multi-modal Models via Stage-wise Visual Context Compression |
|authors| Jieneng ChenLuoxin YeJu HeZhao-Yang WangDaniel KhashabiAlan Yuille
|links| http://arxiv.org/abs/2406.20092v1 |
|updated| 2024-06-28 17:57:14 UTC |
|summary| While significant advancements have been made in compressed representationsfor text embeddings in large language models LLMs the compression of visualtokens in large multi-modal models LMMs has remained a largely overlookedarea. In this work we present the study on the analysis of redundancyconcerning visual tokens and efficient training within these models. Ourinitial experiments show that eliminating up to 70 of visual tokens at thetesting stage by simply average pooling only leads to a minimal 3 reduction invisual question answering accuracy on the GQA benchmark indicating significantredundancy in visual context. Addressing this we introduce Visual ContextCompressor which reduces the number of visual tokens during training toenhance training efficiency without sacrificing performance. To minimizeinformation loss caused by the compression on visual tokens while maintainingtraining efficiency we develop LLaVolta as a lite training scheme. LLaVoltaincorporates stage-wise visual context compression to progressively compressthe visual tokens from heavily to lightly and finally no compression at theend of training yielding no loss of information when testing. Extensiveexperiments demonstrate that our approach enhances the performance of MLLMs inboth image-language and video-language understanding while also significantlycutting training costs. Code is available athttps://github.com/Beckschen/LLaVolta |


| Item |Content|
| --- |---|
|idx| 2406.20085v1 |
|title| Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language |
|authors| Yicheng ChenXiangtai LiYining LiYanhong ZengJianzong WuXiangyu ZhaoKai Chen
|links| http://arxiv.org/abs/2406.20085v1 |
|updated| 2024-06-28 17:53:18 UTC |
|summary| Diffusion-based models have shown great potential in generating high-qualityimages with various layouts which can benefit downstream perception tasks.However a fully automatic layout generation driven only by language and asuitable metric for measuring multiple generated instances has not been wellexplored. In this work we present Auto Cherry-Picker ACP a novel frameworkthat generates high-quality multi-modal training examples to augment perceptionand multi-modal training. Starting with a simple list of natural languageconcepts we prompt large language models LLMs to generate a detaileddescription and design reasonable layouts. Next we use an off-the-shelftext-to-image model to generate multiple images. Then the generated data arerefined using a comprehensively designed metric to ensure quality. Inparticular we present a new metric Composite Layout and Image Score CLISto evaluate the generated images fairly. Our synthetic high-quality examplesboost performance in various scenarios by customizing the initial concept listespecially in addressing challenges associated with long-tailed distributionand imbalanced datasets. Experiment results on downstream tasks demonstratethat Auto Cherry-Picker can significantly improve the performance of existingmodels. In addition we have thoroughly investigated the correlation betweenCLIS and performance gains in downstream tasks and we find that a better CLISscore results in better performance. This finding shows the potential forevaluation metrics as the role for various visual perception and MLLM tasks.Code will be available. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.20088v1 |
|title| Minimax And Adaptive Transfer Learning for Nonparametric Classification under Distributed Differential Privacy Constraints |
|authors| Arnab AuddyT. Tony CaiAbhinav Chakraborty
|links| http://arxiv.org/abs/2406.20088v1 |
|updated| 2024-06-28 17:55:41 UTC |
|summary| This paper considers minimax and adaptive transfer learning for nonparametricclassification under the posterior drift model with distributed differentialprivacy constraints. Our study is conducted within a heterogeneous frameworkencompassing diverse sample sizes varying privacy parameters and dataheterogeneity across different servers. We first establish the minimaxmisclassification rate precisely characterizing the effects of privacyconstraints source samples and target samples on classification accuracy. Theresults reveal interesting phase transition phenomena and highlight theintricate trade-offs between preserving privacy and achieving classificationaccuracy. We then develop a data-driven adaptive classifier that achieves theoptimal rate within a logarithmic factor across a large collection of parameterspaces while satisfying the same set of differential privacy constraints.Simulation studies and real-world data applications further elucidate thetheoretical analysis with numerical results. |


| Item |Content|
| --- |---|
|idx| 2406.20062v1 |
|title| Cost-aware Bayesian optimization via the Pandora's Box Gittins index |
|authors| Qian XieRaul AstudilloPeter FrazierZiv ScullyAlexander Terenin
|links| http://arxiv.org/abs/2406.20062v1 |
|updated| 2024-06-28 17:20:13 UTC |
|summary| Bayesian optimization is a technique for efficiently optimizing unknownfunctions in a black-box manner. To handle practical settings where gatheringdata requires use of finite resources it is desirable to explicitlyincorporate function evaluation costs into Bayesian optimization policies. Tounderstand how to do so we develop a previously-unexplored connection betweencost-aware Bayesian optimization and the Pandoras Box problem a decisionproblem from economics. The Pandoras Box problem admits a Bayesian-optimalsolution based on an expression called the Gittins index which can bereinterpreted as an acquisition function. We study the use of this acquisitionfunction for cost-aware Bayesian optimization and demonstrate empirically thatit performs well particularly in medium-high dimensions. We further show thatthis performance carries over to classical Bayesian optimization withoutexplicit evaluation costs. Our work constitutes a first step towardsintegrating techniques from Gittins index theory into Bayesian optimization. |


| Item |Content|
| --- |---|
|idx| 2406.20044v1 |
|title| Electrostatics-based particle sampling and approximate inference |
|authors| Yongchao Huang
|links| http://arxiv.org/abs/2406.20044v1 |
|updated| 2024-06-28 16:53:06 UTC |
|summary| A new particle-based sampling and approximate inference method based onelectrostatics and Newton mechanics principles is introduced with theoreticalground algorithm design and experimental validation. This method simulates aninteracting particle system IPS where particles i.e. the freely-movingnegative charges and spatially-fixed positive charges with magnitudesproportional to the target distribution interact with each other viaattraction and repulsion induced by the resulting electric fields described byPoissons equation. The IPS evolves towards a steady-state where thedistribution of negative charges conforms to the target distribution. Thisphysics-inspired method offers deterministic gradient-free sampling andinference achieving comparable performance as other particle-based and MCMCmethods in benchmark tasks of inferring complex densities Bayesian logisticregression and dynamical system identification. A discrete-time discrete-spacealgorithmic design readily extendable to continuous time and space isprovided for usage in more general inference problems occurring inprobabilistic machine learning scenarios such as Bayesian inference generativemodelling and beyond. |


| Item |Content|
| --- |---|
|idx| 2406.19958v1 |
|title| The Computational Curse of Big Data for Bayesian Additive Regression Trees: A Hitting Time Analysis |
|authors| Yan Shuo TanOmer RonenTheo SaarinenBin Yu
|links| http://arxiv.org/abs/2406.19958v1 |
|updated| 2024-06-28 14:45:29 UTC |
|summary| Bayesian Additive Regression Trees BART is a popular Bayesiannon-parametric regression model that is commonly used in causal inference andbeyond. Its strong predictive performance is supported by theoreticalguarantees that its posterior distribution concentrates around the trueregression function at optimal rates under various data generative settings andfor appropriate prior choices. In this paper we show that the BART sampleroften converges slowly confirming empirical observations by other researchers.Assuming discrete covariates we show that while the BART posteriorconcentrates on a set comprising all optimal tree structures smallest bias andcomplexity the Markov chains hitting time for this set increases with ntraining sample size under several common data generative settings. As nincreases the approximate BART posterior thus becomes increasingly differentfrom the exact posterior for the same number of MCMC samples contrastingwith earlier concentration results on the exact posterior. This contrast ishighlighted by our simulations showing worsening frequentist undercoverage forapproximate posterior intervals and a growing ratio between the MSE of theapproximate posterior and that obtainable by artificially improving convergencevia averaging multiple sampler chains. Finally based on our theoreticalinsights possibilities are discussed to improve the BART sampler convergenceperformance. |


| Item |Content|
| --- |---|
|idx| 2406.19948v1 |
|title| Kolmogorov-Smirnov GAN |
|authors| Maciej FalkiewiczNaoya TakeishiAlexandros Kalousis
|links| http://arxiv.org/abs/2406.19948v1 |
|updated| 2024-06-28 14:30:14 UTC |
|summary| We propose a novel deep generative model the Kolmogorov-Smirnov GenerativeAdversarial Network KSGAN. Unlike existing approaches KSGAN formulates thelearning process as a minimization of the Kolmogorov-Smirnov KS distancegeneralized to handle multivariate distributions. This distance is calculatedusing the quantile function which acts as the critic in the adversarialtraining process. We formally demonstrate that minimizing the KS distance leadsto the trained approximate distribution aligning with the target distribution.We propose an efficient implementation and evaluate its effectiveness throughexperiments. The results show that KSGAN performs on par with existingadversarial methods exhibiting stability during training resistance to modedropping and collapse and tolerance to variations in hyperparameter settings.Additionally we review the literature on the Generalized KS test and discussthe connections between KSGAN and existing adversarial generative models. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.20087v1 |
|title| ProgressGym: Alignment with a Millennium of Moral Progress |
|authors| Tianyi QiuYang ZhangXuchuan HuangJasmine Xinze LiJiaming JiYaodong Yang
|links| http://arxiv.org/abs/2406.20087v1 |
|updated| 2024-06-28 17:55:24 UTC |
|summary| Frontier AI systems including large language models LLMs hold increasinginfluence over the epistemology of human users. Such influence can reinforceprevailing societal values potentially contributing to the lock-in ofmisguided moral beliefs and consequently the perpetuation of problematicmoral practices on a broad scale. We introduce progress alignment as atechnical solution to mitigate this imminent risk. Progress alignmentalgorithms learn to emulate the mechanics of human moral progress therebyaddressing the susceptibility of existing alignment methods to contemporarymoral blindspots. To empower research in progress alignment we introduceProgressGym an experimental framework allowing the learning of moral progressmechanics from history in order to facilitate future progress in real-worldmoral decisions. Leveraging 9 centuries of historical text and 18 historicalLLMs ProgressGym enables codification of real-world progress alignmentchallenges into concrete benchmarks. Specifically we introduce three corechallenges: tracking evolving values PG-Follow preemptively anticipatingmoral progress PG-Predict and regulating the feedback loop between human andAI value shifts PG-Coevolve. Alignment methods without a temporal dimensionare inapplicable to these tasks. In response we present lifelong andextrapolative algorithms as baseline methods of progress alignment and buildan open leaderboard soliciting novel algorithms and challenges. The frameworkand the leaderboard are available athttps://github.com/PKU-Alignment/ProgressGym andhttps://huggingface.co/spaces/PKU-Alignment/ProgressGym-LeaderBoardrespectively. |


| Item |Content|
| --- |---|
|idx| 2406.19987v1 |
|title| Concept Lens: Visually Analyzing the Consistency of Semantic Manipulation in GANs |
|authors| Sangwon JeongMingwei LiMatthew BergerShusen Liu
|links| http://dx.doi.org/10.1109/VIS54172.2023.00053 |
|updated| 2024-06-28 15:18:40 UTC |
|summary| As applications of generative AI become mainstream it is important tounderstand what generative models are capable of producing and the extent towhich one can predictably control their outputs. In this paper we propose avisualization design named Concept Lens for jointly navigating the datadistribution of a generative model and concept manipulations supported by themodel. Our work is focused on modern vision-based generative adversarialnetworks GAN and their learned latent spaces wherein concept discovery hasgained significant interest as a means of image manipulation. Concept Lens isdesigned to support users in understanding the diversity of a provided set ofconcepts the relationship between concepts and the suitability of concepts togive semantic controls for image generation. Key to our approach is thehierarchical grouping of concepts generated images and the associated jointexploration. We show how Concept Lens can reveal consistent semanticmanipulations for editing images while also serving as a diagnostic tool forstudying the limitations and trade-offs of concept discovery methods. |


| Item |Content|
| --- |---|
|idx| 2406.19954v1 |
|title| BESTOW: Efficient and Streamable Speech Language Model with the Best of Two Worlds in GPT and T5 |
|authors| Zhehuai ChenHe HuangOleksii HrinchukKrishna C. PuvvadaNithin Rao KoluguriPiotr ŻelaskoJagadeesh BalamBoris Ginsburg
|links| http://arxiv.org/abs/2406.19954v1 |
|updated| 2024-06-28 14:40:03 UTC |
|summary| Incorporating speech understanding capabilities into pretrainedlarge-language models has become a vital research direction SpeechLLM. Theprevious architectures can be categorized as: i GPT-style prepend speechprompts to the text prompts as a sequence of LLM inputs like a decoder-onlymodel ii T5-style introduce speech cross-attention to each layer of thepretrained LLMs. We propose BESTOW architecture to bring the BESt features fromTwO Worlds into a single model that is highly efficient and has strongmultitask capabilities. Moreover there is no clear streaming solution foreither style especially considering the solution should generalize to speechmultitask. We reformulate streamable SpeechLLM as a read-write policy problemand unifies the offline and streaming research with BESTOW architecture. Hencewe demonstrate the first open-source SpeechLLM solution that enables Streamingand Multitask at scale beyond ASR at the same time. This streamable solutionachieves very strong performance on a wide range of speech tasks ASR ASTSQA unseen DynamicSuperb. It is end-to-end optimizable with lowertraining/inference cost and demonstrates LLM knowledge transferability tospeech. |


| Item |Content|
| --- |---|
|idx| 2406.19928v1 |
|title| Interactive Topic Models with Optimal Transport |
|authors| Garima DhananiaSheshera MysoreChau Minh PhamMohit IyyerHamed ZamaniAndrew McCallum
|links| http://arxiv.org/abs/2406.19928v1 |
|updated| 2024-06-28 13:57:27 UTC |
|summary| Topic models are widely used to analyze document collections. While they arevaluable for discovering latent topics in a corpus when analysts are unfamiliarwith the corpus analysts also commonly start with an understanding of thecontent present in a corpus. This may be through categories obtained from aninitial pass over the corpus or a desire to analyze the corpus through apredefined set of categories derived from a high level theoretical frameworke.g. political ideology. In these scenarios analysts desire a topic modelingapproach which incorporates their understanding of the corpus while supportingvarious forms of interaction with the model. In this work we present EdTM asan approach for label name supervised topic modeling. EdTM models topicmodeling as an assignment problem while leveraging LM/LLM based document-topicaffinities and using optimal transport for making globally coherenttopic-assignments. In experiments we show the efficacy of our frameworkcompared to few-shot LLM classifiers and topic models based on clustering andLDA. Further we show EdTMs ability to incorporate various forms of analystfeedback and while remaining robust to noisy analyst inputs. |


| Item |Content|
| --- |---|
|idx| 2406.19895v1 |
|title| The Relationship Between Time and Distance Perception in Egocentric and Discrete Virtual Locomotion (Teleportation) |
|authors| Matthias WölwerDaniel Zielasko
|links| http://arxiv.org/abs/2406.19895v1 |
|updated| 2024-06-28 13:03:55 UTC |
|summary| Traveling distances in the real world inherently involves time as moving toa desired location is a continuous process. This temporal component plays arole when estimating the distance covered. However in virtual environmentsthis relationship is often changed or absent. Common teleportation techniquesenable instantaneous transitions lacking any temporal element that might aidin distance perception. Since distances are found to be commonly underestimatedin virtual environments we investigate the influence of time on thismisperception specifically in target-selection-based teleportation interfaces.Our first experiment explores how introducing a delay proportional to thedistance covered by teleportation affects participants perception ofdistances focusing on underestimation accuracy and precision. Participantsare required to teleport along a predefined path with varying delays. A secondexperiment is designed to determine whether this effect manifests in a moreapplication-specific scenario. The results indicate a significant reduction indistance underestimation improving from 27 to 16.8 with a delayedteleportation method. Other sub-scales of distance estimation hardly differ.Despite targeted adaptations of previous study designs participants have againfound strategies supporting them in estimating distances. We conclude that timeis a factor affecting distance perception and should be considered alongsideother factors identified in the literature. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.20041v2 |
|title| BMW Agents -- A Framework For Task Automation Through Multi-Agent Collaboration |
|authors| Noel CrawfordEdward B. DuffyIman EvazzadeTorsten FoehrGregory RobbinsDebbrata Kumar SahaJiya VarmaMarcin Ziolkowski
|links| http://arxiv.org/abs/2406.20041v2 |
|updated| 2024-07-01 16:58:15 UTC |
|summary| Autonomous agents driven by Large Language Models LLMs offer enormouspotential for automation. Early proof of this technology can be found invarious demonstrations of agents solving complex tasks interacting withexternal systems to augment their knowledge and triggering actions. Inparticular workflows involving multiple agents solving complex tasks in acollaborative fashion exemplify their capacity to operate in less strict andless well-defined environments. Thus a multi-agent approach has greatpotential for serving as a backbone in many industrial applications rangingfrom complex knowledge retrieval systems to next generation robotic processautomation. Given the reasoning abilities within the current generation ofLLMs complex processes require a multi-step approach that includes a plan ofwell-defined and modular tasks. Depending on the level of complexity thesetasks can be executed either by a single agent or a group of agents. In thiswork we focus on designing a flexible agent engineering framework with carefulattention to planning and execution capable of handling complex use caseapplications across various domains. The proposed framework providesreliability in industrial applications and presents techniques to ensure ascalable flexible and collaborative workflow for multiple autonomous agentsworking together towards solving tasks. |


| Item |Content|
| --- |---|
|idx| 2406.19930v1 |
|title| Exploring 6G Potential for Industrial Digital Twinning and Swarm Intelligence in Obstacle-Rich |
|authors| Siyu YuanKhurshid AlamBin HanDennis KrummackerHans D. Schotten
|links| http://arxiv.org/abs/2406.19930v1 |
|updated| 2024-06-28 13:57:51 UTC |
|summary| With the advent of 6G technology the demand for efficient and intelligentsystems in industrial applications has surged driving the need for advancedsolutions in target localization. Utilizing swarm robots to locate unknowntargets involves navigating increasingly complex environments. Digital TwinningDT offers a robust solution by creating a virtual replica of the physicalworld which enhances the swarms navigation capabilities. Our frameworkleverages DT and integrates Swarm Intelligence to store physical mapinformation in the cloud enabling robots to efficiently locate unknowntargets. The simulation results demonstrate that the DT framework augmented bySwarm Intelligence significantly improves target location efficiency inobstacle-rich environments compared to traditional methods. This researchunderscores the potential of combining DT and Swarm Intelligence to advance thefield of robotic navigation and target localization in complex industrialsettings. |


| Item |Content|
| --- |---|
|idx| 2406.19852v1 |
|title| FootBots: A Transformer-based Architecture for Motion Prediction in Soccer |
|authors| Guillem CapelleraLuis FerrazAntonio RubioAntonio AgudoFrancesc Moreno-Noguer
|links| http://arxiv.org/abs/2406.19852v1 |
|updated| 2024-06-28 11:49:59 UTC |
|summary| Motion prediction in soccer involves capturing complex dynamics from playerand ball interactions. We present FootBots an encoder-decodertransformer-based architecture addressing motion prediction and conditionedmotion prediction through equivariance properties. FootBots captures temporaland social dynamics using set attention blocks and multi-attention blockdecoder. Our evaluation utilizes two datasets: a real soccer dataset and atailored synthetic one. Insights from the synthetic dataset highlight theeffectiveness of FootBots social attention mechanism and the significance ofconditioned motion prediction. Empirical results on real soccer datademonstrate that FootBots outperforms baselines in motion prediction and excelsin conditioned tasks such as predicting the players based on the ballposition predicting the offensive defensive team based on the ball and thedefensive offensive team and predicting the ball position based on allplayers. Our evaluation connects quantitative and qualitative findings.https://youtu.be/9kaEkfzG3L8 |


| Item |Content|
| --- |---|
|idx| 2406.19742v1 |
|title| Multi-UAVs end-to-end Distributed Trajectory Generation over Point Cloud Data |
|authors| Antonio MarinoClaudio PacchierottiPaolo Robuffo Giordano
|links| http://arxiv.org/abs/2406.19742v1 |
|updated| 2024-06-28 08:29:29 UTC |
|summary| This paper introduces an end-to-end trajectory planning algorithm tailoredfor multi-UAV systems that generates collision-free trajectories inenvironments populated with both static and dynamic obstacles leveraging pointcloud data. Our approach consists of a 2-fork neural network fed with sensingand localization data able to communicate intermediate learned features amongthe agents. One network branch crafts an initial collision-free trajectoryestimate while the other devises a neural collision constraint for subsequentoptimization ensuring trajectory continuity and adherence to physicalactuationlimits. Extensive simulations in challenging cluttered environments involvingup to 25 robots and 25 obstacle density show a collision avoidance successrate in the range of 100 -- 85. Finally we introduce a saliency mapcomputation method acting on the point cloud data offering qualitativeinsights into our methodology. |


| Item |Content|
| --- |---|
|idx| 2406.19477v1 |
|title| Multi-agent Cooperative Games Using Belief Map Assisted Training |
|authors| Qinwei HuangChen LuoAlex B. WuSimon KhanHai LiQinru Qiu
|links| http://dx.doi.org/10.3233/FAIA230444 |
|updated| 2024-06-27 18:40:55 UTC |
|summary| In a multi-agent system agents share their local observations to gain globalsituational awareness for decision making and collaboration using a messagepassing system. When to send a message how to encode a message and how toleverage the received messages directly affect the effectiveness of thecollaboration among agents. When training a multi-agent cooperative game usingreinforcement learning RL the message passing system needs to be optimizedtogether with the agent policies. This consequently increases the modelscomplexity and poses significant challenges to the convergence and performanceof learning. To address this issue we propose the Belief-map AssistedMulti-agent System BAMS which leverages a neuro-symbolic belief map toenhance training. The belief map decodes the agents hidden state to provide asymbolic representation of the agents understanding of the environment andother agents status. The simplicity of symbolic representation allows thegathering and comparison of the ground truth information with the belief whichprovides an additional channel of feedback for the learning. Compared to thesporadic and delayed feedback coming from the reward in RL the feedback fromthe belief map is more consistent and reliable. Agents using BAMS can learn amore effective message passing network to better understand each otherresulting in better performance in a cooperative predator and prey game withvarying levels of map complexity and compare it to previous multi-agent messagepassing models. The simulation results showed that BAMS reduced training epochsby 66 and agents who apply the BAMS model completed the game with 34.62fewer steps on average. |


