import torch
import torch.nn as nn
import pandas as pd
import spacy
from random import randint
from spacy import displacy
from textblob import TextBlob
import numpy as np
from multiprocessing import Process
import nltk
import nltk.data
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, BertEmbeddings
from spacy.gold import biluo_tags_from_offsets
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity as cosinSim

nlp = spacy.load("en_core_web_sm")
embeddings = WordEmbeddings('glove')
EMBEDDING_DIM = 100
document_embeddings = DocumentPoolEmbeddings([embeddings], pooling='max')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english')


def get_token_offset(span, paragraph, answer_start, answer_end):
    p_doc = nlp(paragraph)
    paragraph_tokens = [token.text for token in p_doc]
    span_doc = nlp(span)
    span_tokens = [token.text for token in span_doc]
    [(start, end)] = [(i, i + len(span_tokens)) for i in range(len(paragraph_tokens)) if (
            paragraph_tokens[i] == span_tokens[0] and paragraph_tokens[i:i + len(span_tokens)] == span_tokens)]

    assert paragraph_tokens[start:end] == span_tokens
    return start, end


def token_count(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    return len(tokens)


def sentence_tokenizer(paragraph, tokenizer="textblob"):
    if tokenizer == "spacy":
        doc = nlp(paragraph)
        sentences = [sent.text for sent in doc.sents]
        return sentences, len(sentences)
    elif tokenizer == "nltk":
        sentences = TextBlob(paragraph)
        return sentences, len(sentences)
    else:
        sentences = nltk.tokenize.sent_tokenize(paragraph)
        return sentences, len(sentences)


def glove_cosine_similarity(question, sentence_list):
    question = Sentence(question)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    document_embeddings.embed(question)
    q_emd = question.get_embedding()
    q_emd = q_emd.unsqueeze(0)
    sentence_vectors = torch.empty((1, EMBEDDING_DIM))  # .to(device)
    for idx, sent in enumerate(sentence_list):
        sent = Sentence(sent)
        document_embeddings.embed(sent)
        sent_emd = sent.get_embedding()
        if idx == 0:
            sentence_vectors = sent_emd.unsqueeze(0)
        else:
            sentence_vectors = torch.cat((sentence_vectors, sent_emd.unsqueeze(0)))

    output = cos(q_emd, sentence_vectors)
    return output


def glove_eucleadian(question, sentence_list):
    question = Sentence(question)
    euc = nn.PairwiseDistance(p=2)
    document_embeddings.embed(question)
    q_emd = question.get_embedding()
    q_emd = q_emd.unsqueeze(0)
    sentence_vectors = torch.empty((1, EMBEDDING_DIM))  # .to(device)
    for idx, sent in enumerate(sentence_list):
        sent = Sentence(sent)
        document_embeddings.embed(sent)
        sent_emd = sent.get_embedding()
        if idx == 0:
            sentence_vectors = sent_emd.unsqueeze(0)
        else:
            sentence_vectors = torch.cat((sentence_vectors, sent_emd.unsqueeze(0)))

    output = euc(q_emd, sentence_vectors)
    return output


def generate_target(sentences, start_char_idx):
    target = 1
    tar = 0
    for i in sentences:
        tar = tar + len(i)
        if (tar < start_char_idx):
            target += 1
    return target


def get_tfidf(question, sentences, sim_func):
    index = 0
    documents = [question] + sentences
    tfidf_matrix = tf.fit_transform(documents)
    sims = sim_func(tfidf_matrix[index], tfidf_matrix[index + 1:]).flatten()
    return sims


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def is_root_common(question, sentences):
    q_doc = nlp(question)

    question_nodes = []
    for token in q_doc:
        if token.dep_ == "ROOT":
            question_nodes.append(token.lemma_)
    sentence_contains = []

    for line in sentences:
        flag = 0
        s_doc = nlp(line)
        sent_tokens = []
        for token in s_doc:
            if list(token.children):
                sent_tokens.append(token.lemma_)
        common = intersection(sent_tokens, question_nodes)
        if common:
            flag = 1
        sentence_contains.append(flag)
    return sentence_contains


def fill_missing_dependency(row, max_len, sentences_len):
    if sentences_len < max_len:
        for i in range(sentences_len, max_len):
            row.append(0)
    return row


def fill_missing_cosine(row, max_len, sentences_len):
    replace_value = 0
    if sentences_len < max_len:
        for i in range(sentences_len, max_len):
            row = np.append(row, [replace_value])
    return row


def fill_missing_euclidean(row, max_len, sentences_len):
    replace_value = np.max(row) + np.float(1)
    if sentences_len < max_len:
        for i in range(sentences_len, max_len):
            row = np.append(row, [replace_value])
    return row


def check_features(row, max_len):
    assert len(row["glove_cosine"]) == max_len
    assert len(row["glove_euclidean"]) == max_len
    assert len(row["tfidf_cosine"]) == max_len
    assert len(row["tfidf_euclidean"]) == max_len
    assert len(row["dependency"]) == max_len


def char_offset_to_token_offset_df(data_df):
    counter = 0
    for row in data_df.iterrows():
        index = row[0]
        paragraph = row[1][1]
        span = row[1][2]
        start = row[1][3]
        end = row[1][4]
        # span = paragraph[start:end]

        doc = nlp(paragraph)

        entities = [(start, end, "ANSWER")]

        tags = biluo_tags_from_offsets(doc, entities)

        try:
            if "U-ANSWER" in tags:
                start_tok_idx = tags.index('U-ANSWER')
                end_tok_idx = start_tok_idx
            elif "B-ANSWER" in tags:
                start_tok_idx = tags.index('B-ANSWER')
                end_tok_idx = tags.index('L-ANSWER')
            else:
                continue
            data_df.iloc[index, data_df.columns.get_loc('start_token')] = start_tok_idx
            data_df.iloc[index, data_df.columns.get_loc('end_token')] = end_tok_idx
            counter += 1

            result_span = doc[start_tok_idx:end_tok_idx + 1]
            assert span == str(result_span)
        except Exception as AssertionError:
            continue
    return data_df


def proprocess_csv(in_path, ml_out_path, dl_out_path, min_sent=3, max_sent=10):
    df = pd.read_csv(in_path, header=None, names=["question", "paragraph", "answer_span", "start", "end"])
    df.dropna(inplace=True)
    print("DATA LOADED")
    # df = df[:1000]
    df["q_len_split"] = df["question"].map(lambda question: len(question.split()))
    df["p_len_split"] = df["paragraph"].map(lambda paragraph: len(paragraph.split()))
    df["answer_span_len_split"] = df["answer_span"].map(lambda answer_span: len(str(answer_span).split()))

    df["q_len_spacy"] = df["question"].map(lambda question: token_count(question))
    df["p_len_spacy"] = df["paragraph"].map(lambda paragraph: token_count(paragraph))
    df["answer_span_len_spacy"] = df["answer_span"].map(lambda answer_span: token_count(answer_span))
    #
    df["sentences"] = df["paragraph"].map(lambda paragraph: sentence_tokenizer(paragraph)[0])
    df["sentences_len"] = df["paragraph"].map(lambda paragraph: sentence_tokenizer(paragraph)[1])
    print("LENGTH DONE")

    df = df[df["p_len_split"] <= 156]
    df = df[(df["sentences_len"] >= min_sent) & (df["sentences_len"] <= max_sent)]
    print("REMOVAL DONE")

    df["target"] = df.apply(lambda row: generate_target(row[11], row[3]), axis=1)
    print("TARGET DONE")

    df["glove_cosine"] = df.apply(lambda row: glove_cosine_similarity(row[0], row[11]).tolist(), axis=1)
    print("GLOVE COSINE DONE")

    df["glove_euclidean"] = df.apply(lambda row: glove_eucleadian(row[0], row[11]).tolist(), axis=1)
    print("GLOVE EUCLIDEAN DONE")

    df["tfidf_cosine"] = df.apply(lambda row: get_tfidf(row[0], row[11], linear_kernel), axis=1)
    print("TFIDF COSINE DONE")

    df["tfidf_euclidean"] = df.apply(lambda row: get_tfidf(row[0], row[11], euclidean_distances), axis=1)
    print("TFIDF EUCLIDEAN DONE")

    df["dependency"] = df.apply(lambda row: is_root_common(row[0], row[11]), axis=1)
    print("DEPENDENCY DONE")

    df["glove_cosine"] = df.apply(lambda row: fill_missing_cosine(row[14], max_sent, row[12]), axis=1)
    df["glove_euclidean"] = df.apply(lambda row: fill_missing_euclidean(row[15], max_sent, row[12]), axis=1)
    df["tfidf_cosine"] = df.apply(lambda row: fill_missing_cosine(row[16], max_sent, row[12]), axis=1)
    df["tfidf_euclidean"] = df.apply(lambda row: fill_missing_euclidean(row[17], max_sent, row[12]), axis=1)
    df["dependency"] = df.apply(lambda row: fill_missing_dependency(row[18], max_sent, row[12]), axis=1)

    df.apply(lambda row: check_features(row, max_sent), axis=1)

    df["start_token"] = np.NAN
    df["end_token"] = np.NAN
    df = char_offset_to_token_offset_df(df)
    df = df.dropna()
    df["token_len"] = df.apply(lambda row: row[20] - row[19] + 1, axis=1)

    df["sanity_check"] = df.apply(lambda row: row[21] == row[7], axis=1)

    df = df[df["sanity_check"] == True]
    df = df[(df["answer_span_len_spacy"] < 15)]
    df = df[df["start_token"] < 100]
    df = df[df["end_token"] < 100]

    df["start_token"] = df["start_token"].map(lambda x: int(x))
    df["end_token"] = df["end_token"].map(lambda x: int(x))

    # print("Columns")
    # for idx, i in enumerate(df.columns.values):
    #         print(idx, i)

    dl_data_df = df.copy()
    ml_data_df = df.copy()

    ml_data_df.drop(columns=["question", 'paragraph', 'answer_span', 'start', 'end', 'q_len_split',
                             'p_len_split', 'answer_span_len_split', 'q_len_spacy', 'p_len_spacy',
                             'answer_span_len_spacy', 'sentences', 'sentences_len', "start_token", "end_token",
                             'token_len', 'sanity_check'], inplace=True)

    ml_data_df.to_pickle(ml_out_path)

    dl_data_df.drop(columns=['answer_span', 'start', 'end', 'q_len_split',
                             'p_len_split', 'answer_span_len_split', 'q_len_spacy', 'p_len_spacy',
                             'answer_span_len_spacy', 'sentences', 'sentences_len', 'target',
                             'glove_cosine', 'glove_euclidean', 'tfidf_cosine', 'tfidf_euclidean',
                             'dependency', 'token_len', 'sanity_check'], inplace=True)

    dl_data_df.to_csv(dl_out_path, index=False, header=None)


if __name__ == '__main__':
    print("Preprocesing Train: Started")
    proprocess_csv("./data/train.csv", "./data/preprocess_ml_train.pkl", "./data/preprocess_dl_train.csv")
    print("Preprocesing Train: Completed")

    print("==============================================================================================")

    print("Preprocesing Dev: Started")
    proprocess_csv("./data/dev.csv", "./data/preprocess_ml_dev.pkl", "./data/preprocess_dl_dev.csv")
    print("Preprocesing Dev: Completed")
