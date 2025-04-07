
import re, unidecode, nltk, spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
nlp = spacy.load("es_core_news_sm")



def WordsHistograms(dfs, col1, labels, type_='words', figsize=(12,4)):


    
    fig, ax = plt.subplots(1,1, figsize=figsize)  

    for df, label in zip(dfs, labels):

        if type_=='words':
            sns.histplot(df[col1].str.len(), kde=True, ax=ax, alpha=.4, element="step", label=label)

        elif type_=='tokens':

            # Determining the number of tokens per definition
            df.loc[:, 'num_tokens'] = df[col1].apply(lambda x: nlp(x)).str.len() 
            sns.histplot(df['num_tokens'], kde=True, ax=ax, alpha=.4, element="step", label=label)
    
    plt.title(''' {} Histograms'''.format(col1.capitalize()))
    plt.xlabel('''Number of {}'''.format(type_))
    plt.legend();
    plt.grid()


def WordsHistograms2(df, col1, col2, bins, figsize=(12,4)):


    
    fig, ax = plt.subplots(1,1, figsize=figsize)  
    # Determining the number of tokens per definition
    sns.histplot(df[col1].apply(lambda x: nlp(x)).str.len(), kde=True, ax=ax, alpha=.4, element="step", bins=bins, label=col1.capitalize())
    sns.histplot(df[col2].apply(lambda x: nlp(x)).str.len(), kde=True, ax=ax, alpha=.4, element="step", label=col2.capitalize())    
    plt.title(''' {} and {} Histograms'''.format(col1.capitalize(), col2.capitalize()))
    plt.xlabel('''Number of {}'''.format('tokens'))
    plt.legend();
    plt.grid()



def WordsFrequencies(counts1, counts2, label1, label2, lim_words=20):

    fig, axs = plt.subplots(1,2, figsize=(10,5))  
    
    axs[0].barh(list(counts1.keys())[0:lim_words][::-1], list(counts1.values())[0:lim_words][::-1], color='darkorange')
    axs[0].set_title(label1.capitalize())
    axs[0].set_xlabel('Count')
    axs[0].set_ylabel('Words')
    axs[0].grid()
    axs[1].barh(list(counts2.keys())[0:lim_words][::-1], list(counts2.values())[0:lim_words][::-1], color='royalblue')
    axs[1].set_title(label2.capitalize())
    axs[1].set_xlabel('Count')

    axs[1].grid()
    plt.suptitle('Most Common Words')
    fig.tight_layout()




def EmbeddedWords_Relations(model_wrds, title, figsize=(10,7)):


    vocab = list(model_wrds.wv.key_to_index)
    X = model_wrds.wv[vocab]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    df_w2v = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    df_test = df_w2v.sample(150)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_test['x'], df_test['y'], marker='.', color='darkorange')
    ax.set_xlabel
    ax.grid()
    ax.set_title(title.title())

    for word, pos in df_test.iterrows():
        ax.annotate(word, pos, fontsize=8)
    
