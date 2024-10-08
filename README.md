# 今日から使えるファインチューニングレシピ

<div align="center">
    <img src="./misc/cover.jpg" alt="表紙" width="500">
</div>

本リポジトリは、『[今日から使えるファインチューニングレシピ　－AI・機械学習の技術と実用をつなぐ基本テクニック－](https://www.ohmsha.co.jp/book/9784274232381/)』のコードをまとめたリポジトリです。

## ノートブック

すべてのコードは、Google Colaboratory (Colab) 上で動作確認を行っています。<br>
Chapter 4とChapter 5のコードについては、多くのGPUメモリを必要とするため、Colab Proプラン (有料) にて動作確認を行っています。

| Chapter | 説明 | Colab へのリンク | ファイルへのリンク |
| --- | --- | --- | --- |
| Chapter2 画像のファインチューニング | 2.1 画像分類  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter2/1-1_classification.ipynb) | [リンク](./chapter2/1-1_classification.ipynb) |
|| 2.2 物体検出 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter2/2-1_detection.ipynb) | [リンク](./chapter2/2-1_detection.ipynb) |
|| 2.3 距離学習 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter2/3-1_metric_learning.ipynb) | [リンク](./chapter2/3-1_metric_learning.ipynb) |
|| 2.4 教師無し異常検知 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter2/4-1_anomaly_detection.ipynb) | [リンク](./chapter2/4-1_anomaly_detection.ipynb) |
| Chapter3 自然言語処理のファインチューニング | 3.1 テキスト分類（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/1-1_text_classification_train.ipynb) | [リンク](./chapter3/1-1_text_classification_train.ipynb) |
|   | 3.1 テキスト分類（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/1-2_text_classification_eval.ipynb) | [リンク](./chapter3/1-2_text_classification_eval.ipynb) |
|   | 3.2 マルチラベルテキスト分類（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/2-1_multi_label_classification_train.ipynb) | [リンク](./chapter3/2-1_multi_label_classification_train.ipynb) |
|   | 3.2 マルチラベルテキスト分類（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/2-2_multi_label_classification_eval.ipynb) | [リンク](./chapter3/2-2_multi_label_classification_eval.ipynb) |
|   | 3.3　類似文章検索のファインチューニング（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/3-1_text_similarity_train.ipynb)  | [リンク](./chapter3/3-1_text_similarity_train.ipynb) |
|   | 3.3　類似文章検索のファインチューニング（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter3/3-2_text_similarity_eval.ipynb)  | [リンク](./chapter3/3-2_text_similarity_eval.ipynb) |
| Chapter4 生成AIのファインチューニング | 4.1 プロンプトエンジニアリングによる質問応答 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/1_PromptEngineering.ipynb)  | [リンク](./chapter4/1_PromptEngineering.ipynb) |
|   | 4.2 LoRAによる質問応答（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/2-1_LoRA.ipynb) | [リンク](./chapter4/2-1_LoRA.ipynb) |
|   | 4.2 LoRAによる質問応答（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/2-2_LoRA.ipynb) | [リンク](./chapter4/2-2_LoRA.ipynb) |
|   | 4.3 インストラクションチューニングによる質問応答（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/3-1_InstructionTuning.ipynb) | [リンク](./chapter4/3-1_InstructionTuning.ipynb) |
|   | 4.3 インストラクションチューニングによる質問応答（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/3-2_InstructionTuning.ipynb) | [リンク](./chapter4/3-2_InstructionTuning.ipynb) |
|   | 4.4 画像生成（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/4-1_StableDiffusion.ipynb) | [リンク](./chapter4/4-1_StableDiffusion.ipynb) |
|   | 4.4 画像生成（推論用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter4/4-2_StableDiffusion.ipynb) | [リンク](./chapter4/4-2_StableDiffusion.ipynb) |
| Chapter5 強化学習のファインチューニング |   RLHF（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter5/1-1_rhlf-train.ipynb) | [リンク](./chapter5/1-1_rhlf-train.ipynb) |
|   | RLHF（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter5/1-2_rlhf-eval.ipynb) | [リンク](./chapter5/1-2_rlhf-eval.ipynb) |
|   | 報酬モデルの学習 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter5/1-3_train-reward-model.ipynb) | [リンク](./chapter5/1-3_train-reward-model.ipynb) |
|   | 独自の報酬モデルを用いたRLHF（学習用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter5/1-4_rlhf-train-with-custom-rm.ipynb) | [リンク](./chapter5/1-4_rlhf-train-with-custom-rm.ipynb) |
|   | 独自の報酬モデルを用いたRLHF（評価用） | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laboroai/finetuning_cookbook/blob/main/chapter5/1-5_rlhf-eval.ipynb) | [リンク](./chapter5/1-5_rlhf-eval.ipynb) |


## 正誤情報

本書の正誤情報はオーム社が公開している[正誤表](https://www.ohmsha.co.jp/book/9784274232381/)をご確認ください。

## 参考リンク

* [オーム社のページ](https://www.ohmsha.co.jp/book/9784274232381/)
