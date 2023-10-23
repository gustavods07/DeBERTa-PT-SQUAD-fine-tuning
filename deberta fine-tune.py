from datasets import load_dataset
import datasets
from transformers import Trainer, AutoTokenizer,AutoModelForQuestionAnswering
import pandas as pd
import numpy as np
import torch
import collections
import evaluate
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model_checkpoint = "microsoft/deberta-v3-base"
model_checkpoint = "deepset/deberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
'''
df_train=pd.read_csv('C:/Users/gusta/Documents/FREELA/roBERTa/content/train.csv')
df_test=pd.read_csv('C:/Users/gusta/Documents/FREELA/roBERTa/content/test.csv')
'''



df_train=df_train[['question','human_ans_indices','review','human_ans_spans']]
df_test=df_test[['question','human_ans_indices','review','human_ans_spans']]
     
df_train['id']=np.linspace(0,len(df_train)-1,len(df_train))
df_test['id']=np.linspace(0,len(df_test)-1,len(df_test))

df_train['id']=df_train['id'].astype(str)
df_test['id']=df_test['id'].astype(str)



df_train['answers']=df_train['human_ans_spans']
df_test['answers']=df_test['human_ans_spans']


for i in range(0,len(df_train)):
  answer1={}
  si=int(df_train.iloc[i].human_ans_indices.split('(')[1].split(',')[0])
  ei=int(df_train.iloc[i].human_ans_indices.split('(')[1].split(',')[1].split(' ')[1].split(')')[0])
  answer1['text']=[df_train.iloc[i].review[si:ei]]
  answer1['answer_start']=[si]
  df_train.at[i, 'answers']=answer1
  #print(df_train.iloc[i].answers,df_train.iloc[i].human_ans_spans)

for i in range(0,len(df_test)):
  answer1={}
  si=int(df_test.iloc[i].human_ans_indices.split('(')[1].split(',')[0])
  ei=int(df_test.iloc[i].human_ans_indices.split('(')[1].split(',')[1].split(' ')[1].split(')')[0])
  answer1['text']=[df_test.iloc[i].review[si:ei]]
  answer1['answer_start']=[si]
  df_test.at[i, 'answers']=answer1
  #print(df_train.iloc[i].answers,df_train.iloc[i].human_ans_spans)

df_train.columns


df_train.columns=['question', 'human_ans_indices', 'context', 'human_ans_spans', 'id',
       'answers']

df_test.columns=['question', 'human_ans_indices', 'context', 'human_ans_spans','id',
       'answers']


val_dataset2 = datasets.Dataset.from_pandas(df_test)
train_dataset2 = datasets.Dataset.from_pandas(df_train)


train_dataset = train_dataset2.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_dataset2.column_names,
)
len(train_dataset2), len(train_dataset)



def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


validation_dataset = val_dataset2.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=val_dataset2.column_names,
)

print(validation_dataset)



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

metric = evaluate.load("squad")

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    '''for ind, elem in enumerate(theoretical_answers):
        teorico = theoretical_answers[ind]['answers']['text'][0]
        pratico = answers[ind]['text']
        print(teorico,"-------",pratico)'''
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model.to(device)

from transformers import TrainingArguments

args = TrainingArguments(
    "deberta-finetuned",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    #learning_rate=2e-5,
    #num_train_epochs=5,
    learning_rate=1e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
     
)

n_best=20
max_answer_length = 30

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
metrics = compute_metrics(start_logits, end_logits, validation_dataset, val_dataset2)

print('métricas obtidas pré fine-tuning ')
for metrica in metrics:
    print(metrica)
    print(metrics[metrica])



trainer.train()
trainer.save_model("C:/Users/gusta/Documents/FREELA/roBERTa/my model/")
#trainer.push_to_hub(commit_message="Training complete")

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
metrics2 = compute_metrics(start_logits, end_logits, validation_dataset, val_dataset2)

print('#################################')
print('métricas pós fine-tuning:')
for metrica2 in metrics2:
    print(metrica2)
    print(metrics2[metrica2])


















