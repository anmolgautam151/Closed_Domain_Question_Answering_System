import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json


def train_squad_to_json(filepath, record_path):
    with open(filepath, "r") as file:
        data = json.load(file)
        df1 = json_normalize(data, record_path)
        df2 = json_normalize(data, record_path[:-1])
        df3 = json_normalize(data, record_path[:-2])
        repeated_context = np.repeat(df3['context'].values, df3.qas.str.len())  # repeat for each question
        df2['context'] = repeated_context
        repeated_ids = np.repeat(df2['id'].values, df2['answers'].str.len())  # repeat for each answer
        df1['q_idx'] = repeated_ids
        result_df = df2[['id', 'question', 'context', 'answers',]].set_index('id').reset_index()
        result_df['context_id'] = result_df['context'].factorize()[0]
        result_df['answer_span'] = result_df['answers'].map(
            lambda answers: answers[0]['text'] if len(answers) > 0 else answers)

        result_df['answer_start'] = result_df['answers'].map(
            lambda answers: answers[0]['answer_start'] if len(answers) > 0 else answers)

        result_df['answer_end'] = result_df['answers'].map(
            lambda answers: answers[0]['answer_start'] + len(answers[0]['text']) if len(answers) > 0 else answers)
        result_df['length'] = result_df['answers'].map(lambda answers: len(answers))
        return result_df


def dev_squad_to_json(filepath, record_path):
    with open(filepath, "r") as file:
        data = json.load(file)
        df1 = json_normalize(data, record_path)
        df2 = json_normalize(data, record_path[:-1])
        df3 = json_normalize(data, record_path[:-2])
        repeated_context = np.repeat(df3['context'].values, df3.qas.str.len())  # repeat for each question
        df2['context'] = repeated_context
        repeated_ids = np.repeat(df2['id'].values, df2['answers'].str.len())  # repeat for each answer
        df1['q_idx'] = repeated_ids
        result_df = df2[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
        result_df['context_id'] = result_df['context'].factorize()[0]
        result_df['answer_span'] = result_df['answers'].map(
            lambda answers: answers[0]['text'] if len(answers) > 0 else answers)
        result_df['answer_start'] = result_df['answers'].map(
            lambda answers: answers[0]['answer_start'] if len(answers) > 0 else answers)
        result_df['answer_end'] = result_df['answers'].map(
            lambda answers: answers[0]['answer_start'] + len(answers[0]['text']) if len(answers) > 0 else answers)
        result_df['length'] = result_df['answers'].map(lambda answers: len(answers))
        return result_df

if __name__ == '__main__':
    TRAIN_PATH = ".\\data\\train-v2.0.json"
    DEV_PATH = ".\\data\\dev-v2.0.json"
    record_path = ['data','paragraphs','qas','answers']

    train_df = train_squad_to_json(TRAIN_PATH,record_path)
    train_df.drop(columns=["answers","context_id",'id','length'],inplace=True)
    train_df["answer_start"] =train_df["answer_start"].map(lambda x: -1 if type(x) is list else x) #unknown answer
    train_df["answer_end"] = train_df["answer_end"].map(lambda x: -1 if type(x) is list else x)
    train_df = train_df[train_df.answer_start != -1]
    print("Training Shape",train_df.shape)
    train_df.to_csv(".\\data\\train.csv",sep=",",index=False, header=False)

    dev_df = dev_squad_to_json(DEV_PATH, record_path)
    dev_df.drop(columns=["answers","context_id",'id','length'],inplace=True)
    dev_df["answer_start"] = dev_df["answer_start"].map(lambda x: -1 if type(x) is list else x)
    dev_df["answer_end"] = dev_df["answer_end"].map(lambda x: -1 if type(x) is list else x)
    dev_df = dev_df[dev_df.answer_start!=-1]
    print("Dev Shape",dev_df.shape)
    dev_df.to_csv(".\\data\\dev.csv", sep=",",index=False, header=False)