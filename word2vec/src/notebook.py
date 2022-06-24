


#Set your paths here
jobxml1 = '/content/drive/MyDrive/de-biasing project/Colab/word2vec/data/appointments-job_deduped_n_merged_20180529_100556457696708.xml'
jobxml2 = '/content/drive/MyDrive/de-biasing project/Colab/word2vec/data/benext_com_deduped_n_merged_20180529_101651395833221.xml'

masculinebias = '/content/drive/MyDrive/de-biasing project/Colab/biased-words/masculine_words_suffix.txt'
femininebias = '/content/drive/MyDrive/de-biasing project/Colab/biased-words/feminine_words_suffix.txt'

#These paths can be any
SPECIFYMODELPATH = '/content/w2vmodel.model' #if you use our own generated wiki
SPECIFYWIKIMODELPATH = '/content/w2vwikimodel.wvmodel' #if you download pretrained wiki
SPECIFYJOBADSTXT = '/content/job_ads.txt'


#!!!!!!!!!!!If you use our own generated model, you'll need to copy the model creation thing into the same folder.

import xml.etree.ElementTree as ET
from gensim.utils import simple_preprocess
import nltk
nltk.download('punkt') #Needed before you use it the first time
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


# Extract job descriptions from xml
def extract_job_descriptions(path, xpath):
    tree = ET.parse(path)
    root = tree.getroot()
    return [x.text for x in root.findall(xpath)]

job_ads = extract_job_descriptions(jobxml1, './page/record/jobdescription')
job_ads.extend(extract_job_descriptions(jobxml2, './page/record/job_description'))


# Split into sentences as in the format of training code
def split_to_train():
    with open(SPECIFYJOBADSTXT, 'w') as f:
        for ad in job_ads:
            sent_text = nltk.sent_tokenize(ad) # this gives us a list of sentences
            print(*sent_text, sep='\n', file=f)
            print(file=f)



#Further preprocessing of data by lemmatizing (more powerful version of stemming to obtain the root)
#Also removing stopwords
#Making the words lowercase as well

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
stop_words = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer

wordnet_lemmatizer = WordNetLemmatizer()
regex_tokenizer = RegexpTokenizer(r'\w+')

def clean_sentence(text): #TODO: seems not to lowercase and antipunctualize
    tokenized_text = regex_tokenizer.tokenize(text)
    tokenized_text = [w.lower() for w in tokenized_text if w.isalpha()]
    tokenized_text = [w for w in tokenized_text if not w in stop_words]
    tokenized_text = [wordnet_lemmatizer.lemmatize(w) for w in tokenized_text]
    return (tokenized_text)


def tokenise_ad(job_ad):
    tokens = []
    sent_text = nltk.sent_tokenize(job_ad)
    for line in sent_text:
        tokens.extend(clean_sentence(line.strip()))
    return tokens

tokens = tokenise_ad(job_ads[0])
print(tokens)


import gensim.downloader
def download_pretrained():
    #model = gensim.downloader.load('glove-wiki-gigaword-50') # Download file. ~66MB
    model = gensim.downloader.load('word2vec-google-news-300') # Download file. ~1.666GB
    model.save(SPECIFYWIKIMODELPATH)



import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


#Loading trained model
#w2vmodel = Word2Vec.load(SPECIFYMODELPATH) # loads the model saved in the previous section
#w2vmodelwv = w2vmodel.wv # loads embeddings

#Loading wiki model
w2vmodelwv = KeyedVectors.load(SPECIFYWIKIMODELPATH) # loads wiki model embeddings


#Loading bag of words (for bias)
def load_bag_of_words(path):
    with open(path, 'r') as f:
        return [x.strip() for x in f.readlines()]

masculine = load_bag_of_words(masculinebias)
feminine = load_bag_of_words(femininebias)


#Compute vectors from words. If not found, skip. Prints all not founds if True.
def get_vectors(bag_of_words, print_not_found=True):
    res_vectors = []
    res_words = []
    for word in bag_of_words:
        if word in w2vmodelwv:
            res_vectors.append(w2vmodelwv[word])
            res_words.append(word)
        elif print_not_found:
            print("NOT FOUND: " + word)
    return res_words, res_vectors

m_words, m_vectors = get_vectors(masculine) #existing words and vectors for masculine bias
f_words, f_vectors = get_vectors(feminine)  #existing words and vectors for feminine bias
ad_words, ad_vectors = get_vectors(tokens)  #existing words and vectors for an ad (first in the list)



#Compute trigrams (sum of three vectors)
def get_trigrams(tokens, token_vectors):
    trigram_tokens = []
    trigram_vectors = []
    for i in range(0, len(token_vectors) - 2):
        trigram_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
        trigram_vectors.append(token_vectors[i] + token_vectors[i + 1] + token_vectors[i + 2])

    return trigram_tokens, trigram_vectors

trigrams, trigram_ad_vectors = get_trigrams(ad_words, ad_vectors) #trigrams and vectors for them





#Compute bias score from trigram to bias vectors
def compute_bias_score(ngram_vector, bias_vectors):
    return np.max(KeyedVectors.cosine_similarities(ngram_vector, bias_vectors)) #need to check difference vs similarity

def compute_bias_scores(ngram_vectors, bias_vectors):
    return [compute_bias_score(ngram_vector, bias_vectors) for ngram_vector in ngram_vectors]

ad_m_scores = compute_bias_scores(trigram_ad_vectors, m_vectors) #computed masculine bias scores
ad_f_scores = compute_bias_scores(trigram_ad_vectors, f_vectors) #computed feminine bias scores




import matplotlib.pyplot as plt

# Compute 95th percentile
def get_95_score(raw_scores):
    return np.percentile(raw_scores, 95)

#Compute 95th percentile and plot distribution
def get_plotted_95_score(raw_scores):
    plt.hist(raw_scores)
    plt.axvline(x=np.percentile(raw_scores, 95), color='gray', linestyle='--')
    plt.xlim((-1, 1))
    return get_95_score(raw_scores)

get_plotted_95_score(ad_f_scores)

#Sorting from the highest to the lowest biasness
np.array(trigrams)[np.argsort(ad_m_scores)[::-1]]


def get_ad_score(tokens, bag_of_words):
    bias_words, bias_vectors = get_vectors(bag_of_words, False) #existing words and vectors for some bias
    ad_words, ad_vectors = get_vectors(tokens, False)  #existing words and vectors for an ad (first in the list)

    trigrams, trigram_ad_vectors = get_trigrams(ad_words, ad_vectors) #trigrams and vectors for them

    ad_bias_scores = compute_bias_scores(trigram_ad_vectors, bias_vectors) #computed bias scores

    return get_95_score(ad_bias_scores)


#!!!!!!! (SKIP) For reference: Job ads distribution for scores
# This one will take quite long (time-wise)
def plot_things():
    masculine = load_bag_of_words(masculinebias)
    feminine = load_bag_of_words(femininebias)

    overall_m_scores = [get_ad_score(tokenise_ad(ad), masculine) for ad in job_ads]
    overall_f_scores = [get_ad_score(tokenise_ad(ad), feminine) for ad in job_ads]

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(overall_m_scores)
    axs[1].hist(overall_f_scores)
    plt.setp(axs, xlim=(-1, 1))


#!!!!!!! Plotting graphs of the vector space

#Importing and downloading relevant libraries

import seaborn as sns
from sklearn.manifold import TSNE
#!pip install adjustText
from adjustText import adjust_text
import pandas as pd
import matplotlib.pyplot as plt



def intialise_graphing(model, words):
    # Creating a DataFrame of words in the model and their vectors
    vocab = model.key_to_index;
    vector_list = [model[word] for word in words if word in vocab]
    words_filtered = [word for word in words if word in vocab]
    word_vec_zip = zip(words_filtered, vector_list)
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    return df

def plot_graph(df):
    tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)
    tsne_df = tsne.fit_transform(df[:200])
    sns.set()
    # Initialize figure
    fig, ax = plt.subplots(figsize = (11, 8))
    sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)

    texts = []
    words_to_plot = list(np.arange(0, 200, 10))

    # Append words to list
    for word in words_to_plot:
        texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize = 14))
        
    adjust_text(texts, force_points = 0.4, force_text = 0.4, 
                expand_points = (2,1), expand_text = (1,2),
                arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))

    plt.savefig('vector_space_vis.png')
    plt.show()

plot_graph(intialise_graphing(model, tokens))



def top_ten_biased(trigrams):
    sorted_arr = np.array(trigrams)[np.argsort(ad_m_scores)[::-1]]
    return sorted_arr[:10]



#!!!!!! Further categorising the types of discrimination
""" 
We are using the following stereotypes for age and gender:
Age: assumptions on health and fitness; technological capability; personality, energy and resistance to change
Feminine: traits in common with motherhood; cooperation; lack of assertiveness
Masculine: assumptions on dominance; strength
"""
def age_health(model, word):
    sim_words = model.most_similar('sedentary')
    sim_words.extend(model.most_similar('fragile'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def age_tech(model, word):
    sim_words = model.most_similar('tech')
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def age_personality(model, word):
    sim_words = model.most_similar('dynamic')
    sim_words.extend(model.most_similar('adapt'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def fem_mother(model, word):
    sim_words = model.most_similar('nurture')
    sim_words.extend(model.most_similar('emotional'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def fem_coop(model, word):
    sim_words = model.most_similar('collaboration')
    sim_words.extend(model.most_similar('support'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def fem_gentle(model, word):
    sim_words = model.most_similar('kind')
    sim_words.extend(model.most_similar('gentle'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def masc_dominant(model, word):
    sim_words = model.most_similar('dominate')
    sim_words.extend(model.most_similar('leader'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False

def masc_strong(model, word):
    sim_words = model.most_similar('force')
    sim_words.extend(model.most_similar('independent'))
    for (poss_word, vec) in sim_words:
        if (word == poss_word):
            return True
    return False


#Code to write tags to file

import csv

def tag_words(model, tokens):
    tag_list = []
    for pos, token in enumerate(tokens):
        per_tok_tags = [pos]
        if age_health(model, token):
            per_tok_tags.append("age_health")
        if age_tech(model, token):
            per_tok_tags.append("age_tech")
        if age_personality(model, token):
            per_tok_tags.append("age_personality")
        if fem_mother(model, token):
            per_tok_tags.append("fem_motherhood")
        if fem_coop(model, token):
            per_tok_tags.append("fem_cooperation")
        if fem_gentle(model, token):
            per_tok_tags.append("fem_gentle")
        if masc_dominant(model, token):
            per_tok_tags.append("masc_dominant")
        if masc_strong(model, token):
            per_tok_tags.append("masc_strong")
        tag_list.append(per_tok_tags)
    
    with open("further_tags.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(tag_lst)

tag_words(w2vmodelwv, tokens)

#!!!!!!! Getting synonyms with NLTK WordNet

from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
import nltk.corpus

def get_wordnet_pos(ptag):
    if ptag.startswith('J'):
        return wordnet.ADJ
    elif ptag.startswith('V'):
        return wordnet.VERB
    elif ptag.startswith('N'):
        return wordnet.NOUN
    elif ptag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def get_synonyms():
    ad_text = job_ads[0]
    sentences = nltk.sent_tokenize(ad_text)
    tokens = [nltk.word_tokenize(sent) for sent in sentences]

    nltk.corpus.stopwords.words('english')
    tokens = TreebankWordTokenizer().tokenize(ad_text)
    tokens = nltk.pos_tag(tokens) # adding pos tags
    tokens_ind = [span[0] for span in TreebankWordTokenizer().span_tokenize(ad_text)] # token position in text
    tokens = [(tokens[i][0].lower(), get_wordnet_pos(tokens[i][1]), tokens_ind[i]) for i in range(len(tokens))] # merging + wordnet pos tags + tolower

    tokens = [token for token in tokens if token[0].isalpha()] # filtering non alpha
    tokens = [token for token in tokens if token[0] not in nltk.corpus.stopwords.words('english')] # deleting stopwords
    tokens = [token for token in tokens if token[1] != ''] # deleting all non ADJ VERB NOUN ADV

    def total_bias_scores(m_scores, f_scores):
        return np.array(m_scores) - np.array(f_scores)

    resulting_synonyms = []

    for token in tokens:
        synlist = [synset.lemma_names() for synset in wordnet.synsets(token[0], pos=token[1])]
        synonyms = []
        [synonyms.extend(synl) for synl in synlist]
        synonyms = list(set(synonyms))

        synonyms, syn_vectors = get_vectors(synonyms) #adding the token itself for reference
        syn_bias_scores = compute_bias_scores(syn_vectors, m_vectors)

        t, token_vector = get_vectors([token[0]])

        if len(t) == 0:
            continue

        token_score = compute_bias_scores(token_vector, m_vectors)[0]

        thresh = token_score #getting threshold based on the token itself

        synonyms = np.array(synonyms)[np.argsort(syn_bias_scores)]
        syn_bias_scores = np.sort(syn_bias_scores)

        resulting_synonyms.append((token[0], token[2], synonyms[syn_bias_scores < thresh].tolist()))

    return resulting_synonyms


