import pandas as pd
import json
from sklearn.model_selection import train_test_split

df = pd.DataFrame( columns=['pergunta', 'indices', 'contexto', 'resposta'])

with open('C:/Users/gusta/Documents/FREELA/brquad-gte-dev-v2.0.json',encoding='utf-8') as json_file:
    data = json.load(json_file)
    
    # para cada documento:
    for index_doc,doc in enumerate(data['data']):

        # para cada paragrafo:
        for index_paragraph, paragraph in enumerate(data['data'][index_doc]['paragraphs']):
            context = paragraph['context']

            
            #para cada pergunta
            for index_qas, qas in enumerate(paragraph['qas']):
                question = qas['question']
                id = qas['id']

                #para cada reposta
                for index_resposta, resposta in enumerate(qas['answers']):
                    answer = resposta['text']
                    answer_start = str(resposta['answer_start'])
                    answer_end = str(int(answer_start) + len(answer))
                    human_ans_indices = f"({answer_start}, {answer_end})"

                    # adiciona resposta ao dataframe
                    new_data = {"pergunta": question, "indices": human_ans_indices, "contexto": context, "resposta": answer}
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)


# retirar duplicatas oriiundas do documento JSON
df = df.drop_duplicates()

# embaralhar linhas do dataframe
df = df.sample(frac = 1)


split = len(df)
while split%10 != 0:
    split = split +1

index_threshold = int(split/10 * 7)

train = df.iloc[:index_threshold,:]
test = df.iloc[index_threshold:,:]

#train = df.iloc[:7000,:]
#test = df.iloc[7000:10000,:]


print(len(train),len(test))
train.to_csv('train.csv', encoding='utf-8',index=False)
test.to_csv('test.csv', encoding='utf-8',index=False)

 
