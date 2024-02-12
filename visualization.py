import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('wordnet')


def get_sentence_word_count(text_list):
    """_summary_

    Args:
        text_list (_type_): _description_

    Returns:
        _type_: _description_
    """

    sent_count = 0
    word_count = 0
    vocab = {}

    for text in text_list:
        sentences = sent_tokenize(str(text).lower())
        sent_count = sent_count + len(sentences)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                if word in vocab.keys():
                    vocab[word] = vocab[word] +1
                else:
                    vocab[word] =1 
    word_count = len(vocab.keys())

    return sent_count, word_count


def show_categories(data_categories, title):
    print(f"{'='*10}{title}{'='*10}")
    for i, (cat_name, data_category) in enumerate(data_categories):
        print(f"Cat:{i + 1} {cat_name} : {len(data_category)}")


def count_plot(data_categories):
    plt.figure(figsize=(10,10))
    sns.countplot(y='medical_specialty', data=data_categories)
    plt.show()


def classification_results(labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(cm, annot=True, cmap="Greens", ax=ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
    plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     
    plt.show()

    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
