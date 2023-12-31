# Realizando fine-tuning para responder perguntas com base em um modelo DeBERTa
**DeBERTa fine-tuning (dataset SQUAD em português)**

Os objetivos do projeto são:
- Realizar o fine-tuning de um modelo pré-treinado (microsoft/deberta-v3-base) para responder perguntas com base na arquitetura DeBERTa (Decoding-enhanced BERT with disentangled attention) utilizando um documento SQuAD¹ em portugês do Brasil (traduzido pelo usuário @piEsposito e encontrado em https://github.com/piEsposito/br-quad-2.0).
- Comparar as métricas de exact-match e F1-score do modelo obtido com as métricas de outro modelo fruto de fine-tuning utilizando um documento SQuAD 2.0 em inglês (deepset/deberta-v3-base-squad2).

  
 _¹ (Stanford Question Answering Dataset)_





**Pré-processamento do Dataset**


O documento SQuAD pt-br original possui formatação JSON para encadear as perguntas e suas respectivas respostas de acordo com a estrutura simplificada a seguir:

_DOCUMENTO > CONTEXTOS > PERGUNTAS > RESPOSTAS._


Para facilitar a manipulação dos dados, esse JSON é transformado em uma planilha CSV através do script _json_to_csv.py_ .

A formatação das colunas pode ser observada no exemplo a seguir 
| pergunta| indices | contexto  | resposta | 
| ------------- | ----------------- | ------------- | ------------- |
| "Aonde vivia elias?"  |     (15, 47)      | "Há muito tempo, em uma pequena vila à beira-mar, vivia um pescador solitário chamado Elias"   | "em uma pequena vila à beira-mar"  |


Ao final do script, o dataframe é dividido em duas planilhas, uma para treino e outra para validação, seguindo a proporção 70/30.

**Fine-Tuning**


O script _deberta_fine_tune.py_ foi produzido com adaptações de um artigo de referência² e é responsável por tratar o dataset, realizar o treinamento e comparar as métricas de exact-match e F1-score antes e depois do fine-tuning. Ao final do treinamento, o modelo é salvo localmente e pode ser utilizado via pipeline. A _Tabela 1_ mostra os principais parâmetros para o tokenizer, enquanto a _Tabela 2_ exibe os principais parâmetros para o treinamento.


_²https://medium.com/@xiaohan_63326/fine-tune-fine-tuning-bert-for-question-answering-qa-task-5c29e3d518f1_


_Tabela 1 - Principais parâmetros para o tokenizer._
| Parâmetro| Valor |
| ------------- | ----------------- |
| max_length |384  |
|  truncation| "only_second" |
| stride | 128 |
| return_overflowing_tokens | True |
|  return_offsets_mapping| True |
| padding  | "max_length" |


_Tabela 2 - Principais parâmetros para o treinamento._
| Parâmetro| Valor |
| ------------- | ----------------- |
| evaluation_strategy |"epoch"  |
|  save_strategy| "epoch" |
| learning_rate | 2e-5 |
| num_train_epochs | 5 |
|  weight_decay| 0.01 |
| fp16  | False |





