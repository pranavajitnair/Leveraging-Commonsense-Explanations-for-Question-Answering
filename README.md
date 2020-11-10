# Leveraging Commonsense Explanations for Question Answering
This work has been done as a part of my Natural Language Processing Course Project at Indian Institute of Technology (BHU), Varanasi under the guidence of [Dr. Anil Kumar Singh](https://www.iitbhu.ac.in/dept/cse/people/aksinghcse).

Google's T5 has been used for training on COS-E dataset from [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf). This dataset provides open ended explanations and spans from the questions which may be indicative of the answer for [CommonsenseQA](https://arxiv.org/pdf/1811.00937.pdf) dataset. When these explanations are provided along with context and the questions for CommonsenseQA there is a significant increase in performance. The authors have trained their baseline with GPT

Another dataset used here is the abductive commonsense reasoning dataset [ART](https://arxiv.org/pdf/1908.05739.pdf). Given two observations the task to predict a hypothesis that explains the two observations thus requiring abductive commonsense reasoning. Google T5 has been fine tuned on this dataset. The authors have trained their baseline with GPT-2

|            | COS-E without answers| COS-E with answers| ANLG( ART)| COS-E pretrained on ART| ART pretrained on COS-E| COS-E baseline| ART Baseline|
|------------|----------------------|-------------------|-----------|------------------------|------------------------|---------------|-------------|
|BLEU score  | 7.27                 | 8.445             | 6.19      | 8.265                  | 5.97                   | 4.1           | 3.1         |


## Target task training
We use the above pretrained models to generate explanations for our target task [ReColr](https://openreview.net/pdf?id=HJgJtT4tvB). Commonsense reading comprehension dataset. Given context and question the task is to select the most plausible option. Answering requires commonsense. The contexts are borrowed from previous year GMAT and LCAT papers. Four options are presented to the system for choosing the most plausible one. The task is challegning even for humans( accuracy=63%). The dataset is small moreover the length of the answers are long making the task even more difficult.

I use ALBERT base for all our experiments on ReColr. A significant performance improvement is observed when the explanations are provided to the model along with the context and the question.

|         | Baseline| Pretrainned on COS-E to generate explanations without answers| Pretrained on COS-E to generate explanations with answers| Pretrained on ART to generate explanations|
|---------|---------|--------------------------------------------------------------|----------------------------------------------------------|------------------------------------------|
| Accuracy| 49.8%   | 51.8%                                                        | 51.8%                                                    | 52.0%                                    |
