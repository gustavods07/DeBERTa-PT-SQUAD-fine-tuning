# Realizando fine-tuning para responder perguntas com base em um modelo DeBERTa
**DeBERTa fine-tuning (dataset SQUAD em português)**

O objetivo do projeto é realizar o fine-tuning de um modelo pré-treinado para responder perguntas baseado na arquitetura DeBERTa (Decoding-enhanced BERT with disentangled attention)

O documento SQUAD 2.0 (Stanford Question Answering Dataset) utilizado para fine-tuning foi traduzido pelo usuário @piEsposito e pode ser encontrado em https://github.com/piEsposito/br-quad-2.0

**Pré-processamento do Dataset**


O documento SQUAD original possui formatação JSON para encadear as perguntas e suas respectivas respostas de acordo com a estrutura simplificada a seguir:

_DOCUMENTO > CONTEXTOS > PERGUNTAS > RESPOSTAS._


Para facilitar a manipulação dos dados, esse JSON é transformado em uma planilha CSV através do script _json-to-csv.py_ .

A formatação das colunas pode ser observada a seguir 
| pergunta| indices | contexto  | resposta | 
| ------------- | ----------------- | ------------- | ------------- |
| "Aonde vivia elias"  |     (15, 47)      | "Há muito tempo, em uma pequena vila à beira-mar, vivia um pescador solitário chamado Elias"   | "em uma pequena vila à beira-mar"  |

