import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")

    # Read file and split
def read_paragraph(file_name):
    file = open(file_name, "r")
    ref_text = file.readlines()
    article = ref_text[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences, ref_text

    # Remove stopwords
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [v.lower() for v in sent1]
    sent2 = [v.lower() for v in sent2]

    all_words = list(set(sent1 + sent2))

    vec1 = [0] * len(all_words)
    vec2 = [0] * len(all_words)

    # build the vector for the first sentence
    for v in sent1:
        if v in stopwords:
            continue
        vec1[all_words.index(v)] += 1

    # build the vector for the second sentence
    for v in sent2:
        if v in stopwords:
            continue
        vec2[all_words.index(v)] += 1

    return 1 - cosine_distance(vec1, vec2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    matrix_similarity = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            matrix_similarity[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return matrix_similarity

    # Evaluation Rouge
def evaluate(sum_text, ref_text):
    #print(sum_text, ref_text)
    r = Rouge()
    res = r.get_scores(sum_text, ref_text[0])
    print('Rouge measures recall: ', res)

    # Evaluation Bleu
def bleu(summarize_text):
    candidate = open('candidates.txt', 'r').readlines()

    if len(summarize_text) != len(candidate):
        raise ValueError('The number of sentences in both files do not match.')

    score = 0.

    for i in range(len(summarize_text)):
        score += sentence_bleu([summarize_text[i].strip().split()], candidate[i].strip().split())

    score /= len(summarize_text)
    print("Bleu measures precision: " + str(score))
    pass


def generate_summary(file_name, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # 1st - Read text and split it
    sentences, ref = read_paragraph(file_name)

    # 2nd - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # 3rd - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # 4th - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print("Indexes of top ranked_sentence order are ", ranked_sentence)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    sum_text = '. '.join(summarize_text)
    evaluate(sum_text, ref)
    bleu(summarize_text)

    # 5th - Output the summarize text
    print("Summarize Text: \n", sum_text)


# Summary
generate_summary("text.txt", 4)
