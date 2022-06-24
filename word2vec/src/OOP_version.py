import os
import csv
import pandas as pd
import json
from adjustText import adjust_text
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gensim.downloader
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from gensim.utils import simple_preprocess
import nltk
import time
nltk.download('punkt')  # Needed before you use it the first time
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download("stopwords")
from nltk.corpus import wordnet
#!pip install adjustText

#get root directory by continuously escalating until reaching root
#raises error if not starting in project directory
ROOT = os.getcwd()
while os.path.basename(ROOT) != 'de-biasing-ib-2021':
    ROOT = os.path.dirname(ROOT)
    if ROOT == os.path.dirname(ROOT):
        raise FileNotFoundError("Could not find root directory")

ROOT += '/'

masculinebias = ROOT + 'biased-words/masculine_words_suffix.txt'
femininebias = ROOT + 'biased-words/feminine_words_suffix.txt'

#here, old bias refers to pro old and not anti old, vice versa for young
oldbias = ROOT + 'biased-words/ageism_anti-young.txt'
youngbias = ROOT + 'biased-words/ageism_anti-older.txt'

#These paths can be any
# if you use our own generated wiki
SPECIFYMODELPATH = ROOT + 'word2vec/models/w2vmodel.model'
# if you download pretrained wiki
SPECIFYWIKIMODELPATH = ROOT + 'word2vec/models/w2vwikimodel.model'

SIMWORDSPATH = ROOT + 'word2vec/models/sim_words_dump.json'
SIMTAGSPATH = ROOT + 'word2vec/models/synonym_tags_dump.json'

GRAPHIMGPATH = ROOT + 'vector_space_vis.png'

class Model:

    """
    Args: inp: the job advert fed into the front end as input

    """



    def __init__(self):
        start = time.time()
        self.model = self.download_pretrained()
        end = time.time()
        print(f"Time to load model: {end-start}")


        start = time.time()
        self.bow_masc = self.load_bag_of_words(masculinebias)
        self.bow_fem = self.load_bag_of_words(femininebias)

        self.bow_old = self.load_bag_of_words(oldbias)
        self.bow_young = self.load_bag_of_words(youngbias)
        end = time.time()
        print(f"Time to load bags of words: {end-start}")

        start = time.time()
        self.sim_words = None
        self.tags_to_check = None
        self.load_similar_words()
        end = time.time()
        print(f"Time to load similar tags: {end-start}")



    def compute_ad(self, inp, synonyms_outfile, img_outfile):
        self.synonyms_outfile = synonyms_outfile

        self.inp = inp
        start = time.time()
        self.tokens = self.tokenise_ad(inp)
        end = time.time()
        print(f"Time to tokenize input: {end-start}")

        #TODO: right now, score is just the mean of masc and fem scores, we may want to change this
        start = time.time()
        fem_score  = self.get_ad_score(self.tokens, self.bow_fem)
        masc_score = self.get_ad_score(self.tokens, self.bow_masc)
        self.gender_score = abs(masc_score - fem_score) / 2
        #smoothing
        self.gender_score = -1 * (self.gender_score ** (1.0 / 5.0)) + 1
        end = time.time()
        print(f"Time to calculate gender score: {end-start}")
        #TODO: right now, score is just the mean of old and young scores, we may want to change this
        start = time.time()
        old_score   = self.get_ad_score(self.tokens, self.bow_old)
        young_score = self.get_ad_score(self.tokens, self.bow_young)
        self.age_score = abs(old_score - young_score) / 2
        #smoothing
        self.age_score = -1 * (self.age_score ** (1.0 / 5.0)) + 1
        end = time.time()
        print(f"Time to calculate age score: {end-start}")


        start = time.time()
        self.tagged_words = self.tag_words()
        end = time.time()
        print(f"Time to tag words: {end-start}")

        start = time.time()
        # existing words and vectors for masculine bias
        self.m_words, self.m_vectors = self.get_vectors(self.bow_masc)
        self.f_words, self.f_vectors = self.get_vectors(self.bow_fem)
        self.o_words, self.o_vectors = self.get_vectors(self.bow_old)
        self.y_words, self.y_vectors = self.get_vectors(self.bow_young)
        end = time.time()
        print(f"Time to get masculine/feminine/old/young vectors: {end-start}")

        start = time.time()
        self.synonyms = self.get_synonyms()
        # do not need this anymore
        #self.print_synonyms()
        end = time.time()
        print(f"Time to get synonyms: {end-start}")


        start = time.time()
        #self.graph = self.plot_graph(self.initialise_graphing(), img_outfile)
        end = time.time()
        print(f"Time to get graph: {end-start}")

        return (self.gender_score, fem_score, masc_score, self.age_score, old_score, young_score)


    ##    def extract_job_descriptions(self, path, xpath):
    ##        tree = ET.parse(path)
    ##        root = tree.getroot()
    ##        return [x.text for x in root.findall(xpath)]
    ##
    ##    def split_to_train(self, job_ads):
    ##        with open(SPECIFYJOBADSTXT, 'w') as f:
    ##            for ad in job_ads:
    ##                sent_text = nltk.sent_tokenize(ad) # this gives us a list of sentences
    ##                print(*sent_text, sep='\n', file=f)
    ##                print(file=f)

    def clean_sentence(self, text):
        stop_words = stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        regex_tokenizer = RegexpTokenizer(r'\w+')
        tokenized_text = regex_tokenizer.tokenize(text)
        tokenized_text = [w.lower() for w in tokenized_text if w.isalpha()]
        tokenized_text = [w for w in tokenized_text if not w in stop_words]
        tokenized_text = [wordnet_lemmatizer.lemmatize(
            w) for w in tokenized_text]
        return tokenized_text

    def tokenise_ad(self, job_ad):
        tokens = []
        sent_text = nltk.sent_tokenize(job_ad)
        for line in sent_text:
            tokens.extend(self.clean_sentence(line.strip()))
        return tokens

    def download_pretrained(self):
        model = None
        if not(os.path.isfile(SPECIFYWIKIMODELPATH)) or not(os.path.isfile(SPECIFYWIKIMODELPATH + '.vectors.npy')):
            model = gensim.downloader.load(
                'word2vec-google-news-300')  # Download file. ~1.666GB
            model.save(SPECIFYWIKIMODELPATH)
        else:
            model = KeyedVectors.load(SPECIFYWIKIMODELPATH)
        # i do not know what all these different variables are for
        # there seems to be about 3 different variables for the same thing
        # but i don't want to break anything so keeping them all
        self.w2vmodelwv = model
        return model

    def download_trainedmodel(self):
        # loads the model saved in the previous section
        w2vmodel = Word2Vec.load(SPECIFYMODELPATH)
        self.w2vmodelwv = w2vmodel.wv
        return w2vmodel

    #Loading bag of words (for bias)
    def load_bag_of_words(self, path):
        with open(path, 'r') as f:
            return [x.strip() for x in f.readlines()]

    #Compute vectors from words. If not found, skip. Prints all not founds if True.
    def get_vectors(self, bag_of_words, print_not_found=False):
        res_vectors = []
        res_words = []
        for word in bag_of_words:
            if word in self.w2vmodelwv:
                res_vectors.append(self.w2vmodelwv[word])
                res_words.append(word)
            elif print_not_found:
                print("NOT FOUND: " + word)
        return res_words, res_vectors

    #Compute vectors from words. If not found, skip. Prints all not founds if True.
    def get_phrase_vectors(self, bag_of_words, print_not_found=False):
        res_vectors = []
        res_words = []
        for phrase in bag_of_words:
            phrase_words, phrase_vectors = self.get_vectors(phrase.split("_"))
            if len(phrase_words) == len(phrase.split("_")):
                res_vectors.append(sum(phrase_vectors))
                res_words.append(phrase.replace("_", " "))
            elif print_not_found:
                print("PHRASE NOT FOUND: " + phrase)
        return res_words, res_vectors

    def get_trigrams(self, tokens, token_vectors):
        trigram_tokens = []
        trigram_vectors = []
        for i in range(0, len(token_vectors) - 2):
            trigram_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
            trigram_vectors.append(
                token_vectors[i] + token_vectors[i + 1] + token_vectors[i + 2])

        return trigram_tokens, trigram_vectors

    #Compute bias score from trigram to bias vectors
    def compute_bias_score(self, ngram_vector, bias_vectors):
        # need to check difference vs similarity
        return np.max(KeyedVectors.cosine_similarities(ngram_vector, bias_vectors))

    def compute_bias_scores(self, ngram_vectors, bias_vectors):
        return [self.compute_bias_score(ngram_vector, bias_vectors) for ngram_vector in ngram_vectors]

    # Compute 95th percentile
    def get_95_score(self, raw_scores):
        return np.percentile(raw_scores, 95)

    #Compute 95th percentile and plot distribution
    def get_plotted_95_score(self, raw_scores):
        plt.hist(raw_scores)
        plt.axvline(x=np.percentile(raw_scores, 95),
                    color='gray', linestyle='--')
        plt.xlim((-1, 1))
        return self.get_95_score(raw_scores)

    def get_ad_score(self, tokens, bag_of_words):
        # existing words and vectors for some bias
        bias_words, bias_vectors = self.get_vectors(bag_of_words, False)
        # existing words and vectors for an ad (first in the list)
        ad_words, ad_vectors = self.get_vectors(tokens, False)
        trigrams, trigram_ad_vectors = self.get_trigrams(
            ad_words, ad_vectors)  # trigrams and vectors for them
        ad_bias_scores = self.compute_bias_scores(
            trigram_ad_vectors, bias_vectors)  # computed bias scores
        return self.get_95_score(ad_bias_scores)

    def initialise_graphing(self):
        # Creating a DataFrame of words in the model and their vectors
        vocab = self.model.key_to_index
        vector_list = [self.model[word]
                       for word in self.tokens if word in vocab]
        words_filtered = [word for word in self.tokens if word in vocab]
        word_vec_zip = zip(words_filtered, vector_list)
        word_vec_dict = dict(word_vec_zip)
        df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
        return df

    def plot_graph(self, df, img_outfile):
        tsne = TSNE(n_components=2, init='random',
                    random_state=10, perplexity=100)
        tsne_df = tsne.fit_transform(df[:200])
        sns.set()
        # Initialize figure
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha=0.5)

        texts = []
        words_to_plot = list(np.arange(0, len(df), 1))

        print(len(words_to_plot))
        # Append words to list

        for word in words_to_plot:
            texts.append(
                plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize=14))

        adjust_text(texts, force_points=0.4, force_text=0.4,
                    expand_points=(2, 1), expand_text=(1, 2),
                    arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

        plt.savefig(img_outfile, bbox_inches='tight', pad_inches=0)





    def get_wordnet_pos(self, ptag):
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

    def get_synonyms(self):
        ad_text = self.inp
        sentences = nltk.sent_tokenize(ad_text)
        tokens = [nltk.word_tokenize(sent) for sent in sentences]

        nltk.corpus.stopwords.words('english')
        tokens = TreebankWordTokenizer().tokenize(ad_text)
        tokens = nltk.pos_tag(tokens)  # adding pos tags
        tokens_ind = [span[0] for span in TreebankWordTokenizer(
        ).span_tokenize(ad_text)]  # token position in text
        tokens = [(tokens[i][0].lower(), self.get_wordnet_pos(tokens[i][1]), tokens_ind[i])
                  for i in range(len(tokens))]  # merging + wordnet pos tags + tolower

        # filtering non alpha
        tokens = [token for token in tokens if token[0].isalpha()]
        tokens = [token for token in tokens if token[0]
                  not in nltk.corpus.stopwords.words('english')]  # deleting stopwords
        # deleting all non ADJ VERB NOUN ADV
        tokens = [token for token in tokens if token[1] != '']

        def total_bias_scores(m_scores, f_scores):
            return np.array(m_scores) - np.array(f_scores)


        tag_list = []

        resulting_synonyms = []

        for token in tokens:
            synlist = [synset.lemma_names()
                       for synset in wordnet.synsets(token[0], pos=token[1])]
            synonyms = []
            [synonyms.extend(synl) for synl in synlist]
            synonyms = list(set(synonyms))
            if len(synonyms) == 0:
                continue


            t, token_vector = self.get_vectors([token[0]])

            if len(t) == 0:
                continue

            token_raw_scores = {
                "masc": self.compute_bias_scores(token_vector, self.m_vectors)[0],
                "fem": self.compute_bias_scores(token_vector, self.f_vectors)[0],
                "old": self.compute_bias_scores(token_vector, self.o_vectors)[0],
                "young": self.compute_bias_scores(token_vector, self.y_vectors)[0],
            }

            token_scores = {
                "masc": ((token_raw_scores['masc'] - token_raw_scores['fem']) / 2),
                "fem": ((token_raw_scores['fem'] - token_raw_scores['masc']) / 2),
                "old": ((token_raw_scores['old'] - token_raw_scores['young']) / 2),
                "young": ((token_raw_scores['young'] - token_raw_scores['old']) / 2),
            }


            tag = max(token_scores, key=lambda s: token_scores[s])
            token_score = token_scores[tag]

            def match_tags(tag, further_tag):
                return tag[0:3] == further_tag[0:3] or tag[0:3] in ["old", "you"] and further_tag[0:3] == "age"

            further_tag = ""
            for ftag in self.sim_words.keys():
                if self.find_tag(ftag, token[0]) and match_tags(tag, ftag):
                    further_tag = ftag


            # adding the token itself for reference
            synonyms, syn_vectors = self.get_phrase_vectors(synonyms)
            syn_bias_scores = {}
            syn_bias_scores["masc"] = self.compute_bias_scores(syn_vectors, self.m_vectors)
            syn_bias_scores["fem"] = self.compute_bias_scores(syn_vectors, self.f_vectors)
            syn_bias_scores["old"] = self.compute_bias_scores(syn_vectors, self.o_vectors)
            syn_bias_scores["young"] = self.compute_bias_scores(syn_vectors, self.y_vectors)
            syn_bias_scores["max"] = np.maximum(np.abs(np.array(syn_bias_scores["young"]) - np.array(syn_bias_scores["old"])),
                                                np.abs(np.array(syn_bias_scores["fem"]) - np.array(syn_bias_scores["masc"]))) / 2





            thresh = token_score  # getting threshold based on the token itself

            synonyms = np.array(synonyms)[np.argsort(syn_bias_scores['max'])]
            syn_bias_scores = np.sort(syn_bias_scores['max'])

            per_tok_syns_list = synonyms[syn_bias_scores < thresh].tolist()
            resulting_synonyms.append(
                [(token[0], token_score), token[2], zip(per_tok_syns_list, syn_bias_scores[syn_bias_scores < thresh].tolist())])
            ret_dict = {"synonyms": per_tok_syns_list}
            ret_dict["start"] = token[2]
            ret_dict["end"] = token[2] + len(token[0])
            ret_dict['type'] = tag
            ret_dict['further_type'] = further_tag
            ret_dict["word"] = token[0]

            #convert from np float to python float
            ret_dict['score'] = token_score.item()
            tag_list.append(ret_dict)

        json.dump(tag_list, open(self.synonyms_outfile, 'w+'))

        return resulting_synonyms

    def print_synonyms(self):
        with open(self.synonyms_outfile, 'w') as f:
            for synline in self.synonyms:
                print(synline[0][0], synline[0][1], synline[1], sep=',', end=',', file=f)
                for pa in synline[2]:
                    print(pa[0], pa[1], sep=',', end=",", file=f)
                print("", file=f)

        """ 
    We are using the following stereotypes for age and gender:
    Age: assumptions on health and fitness; technological capability; personality, energy and resistance to change
    Feminine: traits in common with motherhood; cooperation; lack of assertiveness
    Masculine: assumptions on dominance; strength
    """

    def find_tag(self, tag, word):
        for (poss_word, vec) in self.sim_words[tag]:
            if (word == poss_word):
                return True
        return False

    def load_similar_words(self):
        # this function loads a json dump of similar words
        self.sim_words = json.load(open(SIMWORDSPATH, "r"))

        for key, val in self.sim_words.items():
            self.sim_words[key] = [(x[0], x[1]) for x in val]

        self.tags_to_check = self.sim_words.keys()

    def compute_similar_words(self):
        # before we were wasting a lot of time loading these words,
        # so i've changed it so we load them once and then reuse them
        # check if we have loaded before
        # in order for this to really be a good optimisation, we need to get either:
        #   - make the model object work for multiple job inputs
        #   - do some other stuff to make these variables static
        # the first one will lead to way more structured and manageable code
        # but i appreciate we're quite near deadlines if you have to do the second one

        # so this is the one to compute dumps of similar words.
        # in colab the memory used was so huge that it kept crashing
        # so basically this function now only uses normalised vectors
        # (this really screws evth else so should only be used ad hoc)

        self.model.init_sims(replace=True) #that very magical function

        self.sim_words = {}

        self.tags_to_check = ['age_health', 'age_tech', 'age_personality', 'fem_motherhood', 'fem_cooperation', 'fem_gentle', 'masc_dominant', 'masc_strong']

        self.sim_words['age_health'] = self.model.most_similar('sedentary')
        self.sim_words['age_health'].extend(self.model.most_similar('fragile'))
        self.sim_words['age_health'].extend(self.model.most_similar('energy'))
        self.sim_words['age_health'].extend(self.model.most_similar('speed'))

        self.sim_words['age_tech'] = self.model.most_similar('tech')

        self.sim_words['age_personality'] = self.model.most_similar('dynamic')
        self.sim_words['age_personality'].extend(self.model.most_similar('adapt'))
        self.sim_words['age_personality'] = self.model.most_similar('flexible')
        self.sim_words['age_personality'].extend(self.model.most_similar('outgoing'))

        self.sim_words['fem_motherhood'] = self.model.most_similar('nurture')
        self.sim_words['fem_motherhood'].extend(self.model.most_similar('emotional'))
        self.sim_words['fem_motherhood'] = self.model.most_similar('affectionate')
        self.sim_words['fem_motherhood'].extend(self.model.most_similar('tender'))

        self.sim_words['fem_cooperation'] = self.model.most_similar('collaboration')
        self.sim_words['fem_cooperation'].extend(self.model.most_similar('support'))
        self.sim_words['fem_cooperation'] = self.model.most_similar('quiet')
        self.sim_words['fem_cooperation'].extend(self.model.most_similar('submissive'))

        self.sim_words['fem_gentle'] = self.model.most_similar('kind')
        self.sim_words['fem_gentle'].extend(self.model.most_similar('gentle'))
        self.sim_words['fem_gentle'].extend(self.model.most_similar('sympathy'))

        self.sim_words['masc_dominant'] = self.model.most_similar('dominate')
        self.sim_words['masc_dominant'].extend(self.model.most_similar('leader'))
        self.sim_words['masc_dominant'] = self.model.most_similar('courage')
        self.sim_words['masc_dominant'].extend(self.model.most_similar('decisive'))

        self.sim_words['masc_strong'] = self.model.most_similar('force')
        self.sim_words['masc_strong'].extend(self.model.most_similar('independent'))
        self.sim_words['masc_strong'] = self.model.most_similar('assertive')

        json.dump(self.sim_words, open(SIMWORDSPATH, 'w'))

    def tag_words(self):
        tag_list = []

        for pos, token in enumerate(self.tokens):
            per_tok_tags = [pos]
            for tag in self.tags_to_check:
                if self.find_tag(tag, token):
                    per_tok_tags.append(tag)

            tag_list.append(per_tok_tags)

        return tag_list

