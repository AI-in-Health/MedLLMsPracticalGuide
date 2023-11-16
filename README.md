<div align=center>
<img src="img/Medical_LLM_logo.png" width="200px">
</div>
<h2 align="center"> The Practical Guides for Medical Large Language Models </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">


   ![Awesome](https://camo.githubusercontent.com/64f8905651212a80869afbecbf0a9c52a5d1e70beab750dea40a994fa9a9f3c6/68747470733a2f2f617765736f6d652e72652f62616467652e737667)
   [![Arxiv](https://img.shields.io/badge/Arxiv-2311.05112-red)](https://arxiv.org/pdf/2311.05112.pdf)
   [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAI-in-Health%2FMedLLMsPracticalGuide&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</h5>

This is an actively updated list of practical guide resources for medical large language models (LLMs). 
It's based on our survey paper: [A Survey of Large Language Models in Medicine: Progress, Application, and Challenge](https://arxiv.org/abs/2311.05112). **If you want to add your work or model to this list, please do not hesitate to email fenglin.liu@eng.ox.ac.uk and xyzou@uwaterloo.ca.**

## 😮 Highlights

This repository aims to provide an overview of the progress, application, and challenge of LLMs in medicine, aiming to promote further research and exploration in this interdisciplinary field.



##  📣 Update Notes
[2023-11-09] We released the repository.

## Table of Contents
* [Practical Guide for Medical LLMs](#practical-guide-for-medical-llms)
   * [Pre-training from Scratch](#pre-training-from-scratch)
   * [Fine-tuning General LLMs](#fine-tuning-general-llms)
   * [Prompting General LLMs](#prompting-general-llms)
* [Practical Guide for Medical Data](#practical-guide-for-medical-data)
   * [Clinical Knowledge Bases](#clinical-knowledge-bases)
   * [Pre-training Data](#pre-training-data)
   * [Fine-tuning Data](#fine-tuning-data)
* Dowsmtream Biomedical Tasks
   * TODO
* [Practical Guide for Clinical Applications](#practical-guide-for-clinical-applications)
   * [Medical Diagnosis](#medical-diagnosis)
   * [Formatting and ICD Coding](#formatting-and-icd-coding)
   * [Clinical Report Generation](#clinical-report-generation)
   * [Medical Education](#medical-education)
   * [Medical Robotics](#medical-robotics)
   * [Medical Language Translation](#medical-language-translation)
   * [Mental Health Support](#mental-health-support)
* [Practical Guide for Challenges](#practical-guide-for-challenges)
   * [Hallucination](#hallucination)
   * [Lack of Evaluation Benchmarks and Metrics](#lack-of-evaluation-benchmarks-and-metrics)
   * [Domain Data Limitations](#domain-data-limitations)
   * [New Knowledge Adaptation](#new-knowledge-adaptation)
   * [Behavior Alignment](#behavior-alignment)
   * [Ethical, Legal, and Safety Concerns](#ethical-legal-and-safety-concerns)
* [Practical Guide for Future Directions](#practical-guide-for-future-directions)
   * [Introduction of New Benchmarks](#introduction-of-new-benchmarks)
   * [Interdisciplinary Collaborations](#interdisciplinary-collaborations)
   * [Multi-modal LLM](#multi-modal-llm)
   * [LLMs in Less Established Fields of Healthcare](#llms-in-less-established-fields-of-healthcare)


## 🔥 Practical Guide for Medical LLMs

### Pre-training from Scratch

* **BioBERT**: A pre-trained biomedical language representation model for biomedical text mining. 2020. [paper](https://academic.oup.com/bioinformatics/article-abstract/36/4/1234/5566506)
* **PubMedBERT**：Domain-specific language model pretraining for biomedical natural language processing. 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3458754)
* **SciBERT**：A pretrained language model for scientific text. 2019. [paper](https://arxiv.org/abs/1903.10676)
* **ClinicalBERT**：Publicly available clinical BERT embeddings. 2019. [paper](https://arxiv.org/abs/1904.03323)
* **BlueBERT**：Transfer learning in biomedical natural language processing: an evaluation of BERT and ELMo on ten benchmarking datasets. 2019. [paper](https://arxiv.org/abs/1906.05474)
* **BioCPT**：Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. 2023. [paper](https://arxiv.org/abs/2307.00589)
* **BioGPT**：generative pre-trained transformer for biomedical text generation and mining. 2022. [paper](https://academic.oup.com/bib/article-abstract/23/6/bbac409/6713511)
* **OphGLM**：Training an Ophthalmology Large Language-and-Vision Assistant based on Instructions and Dialogue. 2023. [paper](https://arxiv.org/abs/2306.12174)
* **GatorTron**：A large language model for electronic health records. 2022. [paper](https://www.nature.com/articles/s41746-022-00742-2)
* **GatorTronGPT**：A Study of Generative Large Language Model for Medical Research and Healthcare. 2023. [paper](https://arxiv.org/abs/2305.13523)

### Fine-tuning General LLMs

* **DoctorGLM**：Fine-tuning your chinese doctor is not a herculean task. 2023. [paper](https://arxiv.org/abs/2304.01097)
* **BianQue**：Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT. 2023. [paper](https://arxiv.org/abs/2310.15896)
* **ClinicalGPT**：Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation. 2023. [paper](https://arxiv.org/abs/2306.09968)
* **Qilin-Med**：Multi-stage Knowledge Injection Advanced Medical Large Language Model. 2023. [paper](https://arxiv.org/abs/2310.09089)
* **Qilin-Med-VL**：Towards Chinese Large Vision-Language Model for General Healthcare. 2023. [paper](https://arxiv.org/abs/2310.17956)
* **ChatDoctor**：A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge. 2023. [paper](https://www.cureus.com/articles/152858-chatdoctor-a-medical-chat-model-fine-tuned-on-a-large-language-model-meta-ai-llama-using-medical-domain-knowledge.pdf)
* **BenTsao**：Tuning llama model with chinese medical knowledge. 2023. [paper](https://arxiv.org/abs/2304.06975)
* **HuatuoGPT**: HuatuoGPT, towards Taming Language Model to Be a Doctor. 2023. [paper](https://arxiv.org/abs/2305.15075)
* **LLaVA-Med**: Training a large language-and-vision assistant for biomedicine in one day. 2023. [paper](https://arxiv.org/abs/2306.00890)
* **Baize-healthcare**: An open-source chat model with parameter-efficient tuning on self-chat data. 2023. [paper](https://arxiv.org/abs/2304.01196)
* **Visual Med-Alpeca**: A parameter-efficient biomedical llm with visual capabilities. 2023. [Repo](https://github.com/cambridgeltl/visual-med-alpaca)
* **PMC-LLaMA**: Further finetuning llama on medical papers. 2023. [paper](https://arxiv.org/abs/2304.14454)
* **Clinical Camel**: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding. 2023. [paper](https://arxiv.org/abs/2305.12031)
* **MedPaLM 2**: Towards expert-level medical question answering with large language models. 2023. [paper](https://arxiv.org/abs/2305.09617)
* **MedPaLM M**: Towards generalist biomedical ai. 2023. [paper](https://arxiv.org/abs/2307.14334)

### Prompting General LLMs
* **DelD-GPT**: Zero-shot medical text de-identification by gpt-4. 2023. [paper](https://arxiv.org/abs/2303.11032)
* **ChatCAD**: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)
* **Dr. Knows**: Leveraging a medical knowledge graph into large language models for diagnosis prediction. 2023. [paper](https://arxiv.org/abs/2308.14321)
* **MedPaLM**: Large language models encode clinical knowledge. 2022. [paper](https://arxiv.org/abs/2212.13138)


## 📊 Practical Guide for Medical Data

### Clinical Knowledge Bases
* **[Drugs.com](https://www.drugs.com/)**
* **[DrugBank](https://go.drugbank.com/)**
* **[NHS Health](https://www.nhs.uk/conditions/)**
* **[NHS Medicine](https://www.nhs.uk/medicines/)**
* **[Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html)**
* **[The Human Phenotype Ontology](https://hpo.jax.org/app/)**

### Pre-training Data
* **PubMed**: National Institutes of Health. PubMed Data. In National Library of Medicine. 2022. [database](https://pubmed.ncbi.nlm.nih.gov/download/)
* **Literature**: Construction of the literature graph in semantic scholar. 2018. [paper](https://arxiv.org/abs/1805.02262)
* **MIMIC-III**: MIMIC-III, a freely accessible critical care database. 2016. [paper](https://www.nature.com/articles/sdata201635)
* **PubMed**: The pile: An 800gb dataset of diverse text for language modeling. 2020. [paper](https://arxiv.org/abs/2101.00027)
* **MedDialog**: Meddialog: Two large-scale medical dialogue datasets. 2020. [paper](https://arxiv.org/abs/2004.03329)
* **EHRs**: A large language model for electronic health records. 2022. [paper](https://www.nature.com/articles/s41746-022-00742-2)
* **EHRs**: A Study of Generative Large Language Model for Medical Research and Healthcare. 2023. [paper](https://arxiv.org/abs/2305.13523)

### Fine-tuning Data
* **CMD.**: Chinese medical dialogue data. 2023. [repo](https://github.com/Toyhom/Chinese-medical-dialogue-data)
* **BianQueCorpus**: BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT. 2023. [paper](https://arxiv.org/abs/2310.15896)
* **MD-EHR**: ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation. 2023. [paper](https://arxiv.org/abs/2306.09968)
* **VariousMedQA**: Multi-scale attentive interaction networks for chinese medical question answer selection. 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8548603/)
* **VariousMedQA**: What disease does this patient have? a large-scale open domain question answering dataset from medical exams. 2021. [paper](https://www.mdpi.com/2076-3417/11/14/6421)
* **MedDialog**: Meddialog: Two large-scale medical dialogue datasets. 2020. [paper](https://arxiv.org/abs/2004.03329)
* **ChiMed**: Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model. 2023. [paper](https://arxiv.org/abs/2310.09089)
* **ChiMed-VL**: Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare. 2023. [paper](https://arxiv.org/abs/2310.17956)
* **Healthcare Magic**: Healthcare Magic. [platform](https://www.healthcaremagic.com/)
* **ICliniq**: ICliniq. [platform](https://www.icliniq.com/)
* **Hybrid SFT**: HuatuoGPT, towards Taming Language Model to Be a Doctor. 2023. [paper](https://arxiv.org/abs/2305.15075)
* **PMC-15M**: Large-scale domain-specific pretraining for biomedical vision-language processing. 2023. [paper](https://arxiv.org/abs/2303.00915)
* **MedQuAD**: A question-entailment approach to question answering. 2019. [paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4?ref=https://githubhelp.com)
* **VariousMedQA**: Visual med-alpaca: A parameter-efficient biomedical llm with
visual capabilities. 2023. [repo](https://github.com/cambridgeltl/visual-med-alpaca)
* **MTB**: Med-flamingo: a multimodal medical few-shot learner. 2023. [paper](https://arxiv.org/abs/2307.15189)
* **PMC-OA**: Pmc-clip: Contrastive language-image pre-training using biomedical documents. 2023. [paper](https://arxiv.org/abs/2303.07240)
* **Medical Meadow**: MedAlpaca--An Open-Source Collection of Medical Conversational AI Models and Training Data. 2023. [paper](https://arxiv.org/abs/2304.08247)
* **Literature**: S2ORC: The semantic scholar open research corpus. 2019. [paper](https://arxiv.org/abs/1911.02782)
* **MedC-I**: Pmc-llama: Further finetuning llama on medical papers. 2023. [paper](https://arxiv.org/abs/2304.14454)
* **ShareGPT**: Sharegpt. 2023. [platform](https://sharegpt.com/)
* **PubMed**: National Institutes of Health. PubMed Data. In National Library of Medicine. 2022. [database](https://pubmed.ncbi.nlm.nih.gov/download/)
* **MedQA**: What disease does this patient have? a large-scale open domain question answering dataset from medical exams. 2021. [paper](https://www.mdpi.com/2076-3417/11/14/6421)
* **MultiMedQA**: Towards expert-level medical question answering with large language models. 2023. [paper](https://arxiv.org/abs/2305.09617)
* **MultiMedBench**: Towards generalist biomedical ai. 2023. [paper](https://arxiv.org/abs/2307.14334)

## ✨ Practical Guide for Clinical Applications

### Medical Diagnosis
* Designing a Deep Learning-Driven Resource-Efficient Diagnostic System for Metastatic Breast Cancer: Reducing Long Delays of Clinical Diagnosis and Improving Patient Survival in Developing Countries. 2023. [paper](https://arxiv.org/abs/2308.02597)
* AI in health and medicine. 2022. [paper](https://www.nature.com/articles/s41591-021-01614-0)
* Large language models in medicine. 2023. [paper](https://www.nature.com/articles/s41591-023-02448-8)
* Leveraging a medical knowledge graph into large language models for diagnosis prediction. 2023. [paper](https://arxiv.org/abs/2308.14321)
* Chatcad: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)

### Formatting and ICD-Coding
* Applying large language model artificial intelligence for retina International Classification of Diseases (ICD) coding. 2023. [paper](https://jmai.amegroups.org/article/view/8198/html)
* PLM-ICD: automatic ICD coding with pretrained language models. 2022. [paper](https://arxiv.org/abs/2207.05289)

### Clinical Report Generation
* Using ChatGPT to write patient clinic letters. 2023. [paper](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00048-1/fulltext)
* ChatGPT: the future of discharge summaries?. 2023. [paper](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00021-3/fulltext)
* Chatcad: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)
* Can GPT-4V (ision) Serve Medical Applications? Case Studies on GPT-4V for Multimodal Medical Diagnosis. 2023. [paper](https://arxiv.org/abs/2310.09909)
* Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare. 2023. [paper](https://arxiv.org/abs/2310.17956)
* Customizing General-Purpose Foundation Models for Medical Report Generation. 2023. [paper](https://arxiv.org/abs/2306.05642)
* Towards generalist foundation model for radiology. 2023. [paper](https://arxiv.org/abs/2308.02463)
* Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts. 2023. [paper](https://arxiv.org/abs/2309.07430)

### Medical Education
* Large Language Models in Medical Education: Opportunities, Challenges, and Future Directions. 2023. [paper](https://mededu.jmir.org/2023/1/e48291/)
* The Advent of Generative Language Models in Medical Education. 2023. [paper](https://mededu.jmir.org/2023/1/e48163)
* The impending impacts of large language models on medical education. 2023. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10020064/)

### Medical Robotics
* A Nested U-Structure for Instrument Segmentation in Robotic Surgery. 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10218893/)
* The multi-trip autonomous mobile robot scheduling problem with time windows in a stochastic environment at smart hospitals. 2023. [paper](https://www.mdpi.com/2076-3417/13/17/9879)
* Advanced robotics for medical rehabilitation. 2016. [paper](https://link.springer.com/content/pdf/10.1007/978-3-319-19896-5.pdf)
* GRID: Scene-Graph-based Instruction-driven Robotic Task Planning. 2023. [paper](https://arxiv.org/abs/2309.07726)
* Trust in Construction AI-Powered Collaborative Robots: A Qualitative Empirical Analysis. 2023. [paper](https://arxiv.org/abs/2308.14846)

### Medical Language Translation
* Machine translation of standardised medical terminology using natural language processing: A Scoping Review. 2023. [paper](https://www.sciencedirect.com/science/article/pii/S1871678423000432)
* The Advent of Generative Language Models in Medical Education. 2023. [paper](https://mededu.jmir.org/2023/1/e48163)
* The impending impacts of large language models on medical education. 2023. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10020064/)

### Mental Health Support
* ChatCounselor: A Large Language Models for Mental Health Support. 2023. [paper](https://arxiv.org/abs/2309.15461)
* Tell me, what are you most afraid of? Exploring the Effects of Agent Representation on Information Disclosure in Human-Chatbot Interaction, 2023, [paper](https://link.springer.com/chapter/10.1007/978-3-031-35894-4_13)
* A Brief Wellbeing Training Session Delivered by a Humanoid Social Robot: A Pilot Randomized Controlled Trial. 2023. [paper](https://link.springer.com/article/10.1007/s12369-023-01054-5)
* Real conversations with artificial intelligence: A comparison between human–human online conversations and human–chatbot conversations. 2015. [paper](https://www.sciencedirect.com/science/article/pii/S0747563215001247)

## ⚔️ Practical Guide for Challenges

### Hallucination
* Survey of hallucination in natural language generation. 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3571730)
* Med-halt: Medical domain hallucination test for large language models. 2023. [paper](https://arxiv.org/abs/2307.15343)
* A survey of hallucination in large foundation models. 2023. [paper](https://arxiv.org/abs/2309.05922)
* Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. 2023. [paper](https://arxiv.org/abs/2303.08896)
* Retrieval augmentation reduces hallucination in conversation. 2021. [paper](https://arxiv.org/abs/2104.07567)
* Chain-of-verification reduces hallucination in large language models. 2023. [paper](https://arxiv.org/abs/2309.11495)

### Lack of Evaluation Benchmarks and Metrics
* What disease does this patient have? a large-scale open domain question answering dataset from medical exams. 2021. [paper](https://www.mdpi.com/2076-3417/11/14/6421)
* Truthfulqa: Measuring how models mimic human falsehoods. 2021. [paper](https://arxiv.org/abs/2109.07958)
* HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. 2023. [paper](https://ui.adsabs.harvard.edu/abs/2023arXiv230511747L/abstract)

### Domain Data Limitations
* Textbooks Are All You Need. 2023. [paper](https://arxiv.org/abs/2306.11644)
* Model Dementia: Generated Data Makes Models Forget. 2023. [paper](https://arxiv.org/abs/2305.17493)

### New Knowledge Adaptation
* Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark. 2023. [paper](https://arxiv.org/abs/2305.17553)
* Editing Large Language Models: Problems, Methods, and Opportunities. 2023. [paper](https://arxiv.org/abs/2305.13172)
* Retrieval-augmented generation for knowledge-intensive nlp tasks. 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

### Behavior Alignment
* Aligning ai with shared human values. 2020. [paper](https://arxiv.org/abs/2008.02275)
* Training a helpful and harmless assistant with reinforcement learning from human feedback. 2022. [paper](https://arxiv.org/abs/2204.05862)
* Improving alignment of dialogue agents via targeted human judgements. 2022. [paper](https://arxiv.org/abs/2209.14375)
* Webgpt: Browser-assisted question-answering with human feedback. 2021. [paper](https://arxiv.org/abs/2112.09332)
* Languages are rewards: Hindsight finetuning using human feedback. 2023. [paper](https://arxiv.org/abs/2302.02676)

### Ethical, Legal, and Safety Concerns
* ChatGPT utility in healthcare education, research, and practice: systematic review on the promising perspectives and valid concerns. 2023. [paper](https://www.mdpi.com/2227-9032/11/6/887)
* ChatGPT listed as author on research papers: many scientists disapprove. 2023. [paper](https://ui.adsabs.harvard.edu/abs/2023Natur.613..620S/abstract)
* A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics. 2023. [paper](https://arxiv.org/abs/2310.05694)
* Multi-step jailbreaking privacy attacks on chatgpt. 2023. [paper](https://arxiv.org/abs/2304.05197)
* Jailbroken: How does llm safety training fail?. 2023. [paper](https://arxiv.org/abs/2307.02483)

## 🚀 Practical Guide for Future Directions


### Introduction of New Benchmarks
* A comprehensive benchmark study on biomedical text generation and mining with ChatGPT. 2023. [paper](https://www.biorxiv.org/content/10.1101/2023.04.19.537463.abstract)
* Creation and adoption of large language models in medicine. 2023. [paper](https://jamanetwork.com/journals/jama/article-abstract/2808296)

### Interdisciplinary Collaborations
* Creation and adoption of large language models in medicine. 2023. [paper](https://jamanetwork.com/journals/jama/article-abstract/2808296)
* ChatGPT and Physicians' Malpractice Risk. 2023. [paper](https://jamanetwork.com/journals/jama-health-forum/fullarticle/2805334)

### Multi-modal LLM
* A Survey on Multimodal Large Language Models. 2023. [paper](https://arxiv.org/abs/2306.13549)
* Mm-react: Prompting chatgpt for multimodal reasoning and action. 2023. [paper](https://arxiv.org/abs/2303.11381)
* ChatGPT for shaping the future of dentistry: the potential of multi-modal large language model. 2023. [paper](https://www.nature.com/articles/s41368-023-00239-y)
* Frozen Language Model Helps ECG Zero-Shot Learning. 2023. [paper](https://arxiv.org/abs/2303.12311)
* Exploring and Characterizing Large Language Models For Embedded System Development and Debugging. 2023. [paper](https://arxiv.org/abs/2307.03817)
* MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. 2023. [paper](https://arxiv.org/abs/2306.13394)

### LLMs in less established fields of healthcare
* Large Language Models in Sport Science & Medicine: Opportunities, Risks and Considerations. 2023. [paper](https://arxiv.org/abs/2305.03851)

## 👍 Acknowledgement
* [LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide). The codebase we built upon and it is a comprehensive LLM suvey.


## Citation

Please consider citing our papers if our repository is helpful to your work, thanks sincerely!

```bibtex
@article{zhou2023survey,
   title={A Survey of Large Language Models in Medicine: Progress, Application, and Challenge},
   author={Hongjian Zhou, Boyang Gu, Xinyu Zou, Yiru Li, Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Xian Wu, Zheng Li, Fenglin Liu},
   journal={arXiv preprint 2311.05112}
   year={2023}
}
```


