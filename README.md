# Realizando fine-tuning para responder perguntas com base em um modelo DeBERTa
**DeBERTa fine-tuning (dataset SQUAD em português)**

O objetivo do projeto é realizar o fine-tuning de um modelo pré-treinado (microsoft/deberta-v3-base) para responder perguntas com base na arquitetura DeBERTa (Decoding-enhanced BERT with disentangled attention) utilizando um documento SQuAD¹ em portugês do Brasil (traduzido pelo usuário @piEsposito e encontrado em https://github.com/piEsposito/br-quad-2.0) para comparar as métricas de exact-match e F1-score com outro modelo (deepset/deberta-v3-base-squad2), fruto de fine-tuning utilizando um documento SQuAD 2.0 em inglês.

 _¹(Stanford Question Answering Dataset)_

**Pré-processamento do Dataset**


O documento SQuAD pt-br original possui formatação JSON para encadear as perguntas e suas respectivas respostas de acordo com a estrutura simplificada a seguir:

_DOCUMENTO > CONTEXTOS > PERGUNTAS > RESPOSTAS._


Para facilitar a manipulação dos dados, esse JSON é transformado em uma planilha CSV através do script _json_to_csv.py_ .

A formatação das colunas pode ser observada no exemplo a seguir 
| pergunta| indices | contexto  | resposta | 
| ------------- | ----------------- | ------------- | ------------- |
| "Aonde vivia elias?"  |     (15, 47)      | "Há muito tempo, em uma pequena vila à beira-mar, vivia um pescador solitário chamado Elias"   | "em uma pequena vila à beira-mar"  |


Ao final do script, o dataframe é dividido em duas planilhas, uma para treino e outra para validação, seguindo a proporção 70/30.

