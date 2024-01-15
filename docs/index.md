| Item |Content|
| --- |---|
|idx| 1 |
|title| Machine Translation Models are Zero-Shot Detectors of Translation Direction |
|authors| Michelle WastlJannis VamvasRico Sennrich
|links| http://arxiv.org/abs/2401.06769v1 |
|updated| 2024-01-12 18:59:02 UTC |
|summary| Detecting the translation direction of parallel text has applications formachine translation training and evaluation, but also has forensic applicationssuch as resolving plagiarism or forgery allegations. In this work, we explorean unsupervised approach to translation direction detection based on the simplehypothesis that$p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$,motivated by the well-known simplification effect in translationese ormachine-translationese. In experiments with massively multilingual machinetranslation models across 20 translation directions, we confirm theeffectiveness of the approach for high-resource language pairs, achievingdocument-level accuracies of 82-96% for NMT-produced translations, and 60-81%for human translations, depending on the model used. Code and demo areavailable at https://github.com/ZurichNLP/translation-direction-detection |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements |
|authors| Anton VoronovLena WolfMax Ryabinin
|links| http://arxiv.org/abs/2401.06766v1 |
|updated| 2024-01-12 18:58:26 UTC |
|summary| Large language models demonstrate a remarkable capability for learning tosolve new tasks from a few examples. The prompt template, or the way the inputexamples are formatted to obtain the prompt, is an important yet oftenoverlooked aspect of in-context learning. In this work, we conduct acomprehensive study of the template format's influence on the in-contextlearning performance. We evaluate the impact of the prompt template acrossmodels (from 770M to 70B parameters) and 4 standard classification datasets. Weshow that a poor choice of the template can reduce the performance of thestrongest models and inference methods to a random guess level. Moreimportantly, the best templates do not transfer between different setups andeven between models of the same family. Our findings show that the currentlyprevalent approach to evaluation, which ignores template selection, may givemisleading results due to different templates in different works. As a firststep towards mitigating this issue, we propose Template Ensembles thataggregate model predictions across several templates. This simple test-timeaugmentation boosts average performance while being robust to the choice ofrandom set of templates. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding |
|authors| Mingdao LiuAohan ZengBowen WangPeng ZhangJie TangYuxiao Dong
|links| http://arxiv.org/abs/2401.06761v1 |
|updated| 2024-01-12 18:50:36 UTC |
|summary| The massive adoption of large language models (LLMs) demands efficientdeployment strategies. However, the auto-regressive decoding process, which isfundamental to how most LLMs generate text, poses challenges to achieveefficient serving. In this work, we introduce a parallel auto-regressivegeneration method. By instruct-tuning on general domain data that containshierarchical structures, we enable LLMs to independently plan their generationprocess and perform auto-parallel auto-regressive (APAR) generation,significantly reducing the number of generation steps. APAR alone can achieveup to 2x speed-up, and when combined with speculative decoding, the speed-upcan reach up to 4x. In addition, APAR reduces the key-value cache consumptionand attention computation during generation. This leads to a throughputincrease of 20-70% and a latency reduce of 20-35% in high-throughput scenarios,compared to state-of-the-art serving frameworks. |


