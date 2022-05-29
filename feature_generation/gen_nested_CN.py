import csv
import pickle
from collections import defaultdict
import spacy
from nltk.stem import PorterStemmer
import argparse
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()


def get_singular(token):
    # returns the singular form of a token, if not plural or no singular form of plural: return False
    token1 = nlp(token)
    lemma_tags_plural = {"NNS", "NNPS"}

    for token in token1:
        lemma = False
        if token.tag_ in lemma_tags_plural:
            lemma_temp = token.lemma_
            if str(lemma_temp) != str(token):
                lemma = lemma_temp
        return lemma

def get_plurals(openwords_set):
    """ returns:
        total_plural = all in top10th that are plural
        words_sing = the singular of each of the total_plural  words
        """
    # store all the words that are plural
    total_plural = []
    for token in openwords_set:
        lemma = get_singular(token)
        if lemma:
            total_plural.append(token)

    # create a lookup dict to go from singular to plural            
    words_sing = []  # list of all the single words whose plurals are in sing_to_plur
    sing_to_plur = dict()
    for plur_word in total_plural:
        word_sing = get_singular(plur_word)
        if not word_sing:
            print("ERROR there is no singular word")
        if word_sing in words_sing:
            print("singular already exists",word_sing, plur_word, sing_to_plur[word_sing])
        sing_to_plur[word_sing] = plur_word
        words_sing.append(word_sing)
    return total_plural, sing_to_plur, words_sing

def remove_duplicate_rc(lookupdict):
    lookupdict2 = dict()
    for k ,v in lookupdict.items():
        stored_rel_concepts = []
        stored_items = []
        for item in v:
            rc_idx = int(not(item[-1]))
            rel_concept = item[rc_idx]
            if rel_concept in stored_rel_concepts:
                continue
            stored_rel_concepts.append(rel_concept)
            stored_items.append(item)
        lookupdict2[k] = stored_items
    return lookupdict2


def parse_rel_label(lookupdict):
    lookupdict2 = lookupdict.copy()
    for k,v in lookupdict2.items():
        for relitem in v:
            relidx = relitem[-1]
            relationlabel = relitem[relidx]
            newrellabel = "<|"+ relationlabel+"|>"
            relitem[relidx] = newrellabel
    return lookupdict2


def main(openweb10th_path, ConcNet_Eng_path, out_path, args):
    f = open(openweb10th_path, 'r', encoding="UTF-8")
    with f:
        reader = csv.reader(f)
        all_openwebwords = [word[0] for word in reader]
    openwords_set = set(all_openwebwords)
    print("number of keywords from file:", len(all_openwebwords))
    stopwords = nlp.Defaults.stop_words  # load all stopwords


    total_plural, sing_to_plur, words_sing = get_plurals(openwords_set)
    concNet_file = open(ConcNet_Eng_path)
    csvreader_list = list(csv.reader(concNet_file))
    tot_csv = len(csvreader_list)
    # concNet_nested_filt = defaultdict(list)

    concNet_nested_filt = {k:[] for k in all_openwebwords}
    for i in tqdm(range(tot_csv)):
        row = csvreader_list[i]
        conc_one = str(row[2].split("/")[3]).replace("_", " ")
        conc_two = str(row[3].split("/")[3]).replace("_", " ")
        rel = str(row[1].split("/")[-1])
        if conc_one in stopwords or conc_two in stopwords:
            continue
        if conc_one in openwords_set and conc_two in openwords_set:
            continue 

        stem1 = ps.stem(conc_one)
        stem2 = ps.stem(conc_two)
        if stem1 == stem2:
            continue

        plur1 = sing_to_plur[conc_one] if conc_one in words_sing else conc_one
        plur2 = sing_to_plur[conc_two] if conc_two in words_sing else conc_two

        # if concone is bird, and plural is birds, an : append idx to birds
        if plur1 in total_plural and plur1 != conc_one and conc_two not in openwords_set:
            concNet_nested_filt[plur1].append([rel, conc_two, 0])
        if plur2 in total_plural and plur2 != conc_two and conc_one not in openwords_set:
            concNet_nested_filt[plur2].append([conc_one, rel, 1])

        #was here
        if conc_one in openwords_set:
            concNet_nested_filt[conc_one].append([rel, conc_two, 0])
        elif conc_two in openwords_set:
            concNet_nested_filt[conc_two].append([conc_one, rel, 1])
    
    print("Finished creating the filtered nested dict")
    concNet_nested_filt0 = concNet_nested_filt.copy()

    if not args.no_parse_rel_label:
        # parse the relationshiplabel
        concNet_nested_filt1 = parse_rel_label(concNet_nested_filt0)
        print("Finished parsing relationship labels")

    if not args.allow_duplicate:
        # remove duplicate related concepts
        concNet_nested_filt = remove_duplicate_rc(concNet_nested_filt1)
        print("Finished removing duplicates")


    file_to_store = open(out_path, "wb")
    pickle.dump(concNet_nested_filt, file_to_store)
    file_to_store.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openweb10th_path', default="../../data_files/openwebtext_10th.csv", type=str)
    parser.add_argument('--ConcNet_Eng_path', default="../../data_files/ass_onlyenglish.csv", type=str)
    parser.add_argument('--out_path', default="../../data_files/CN_feats/conceptnet_filt_nest_labels_new.pkl", type=str)

    parser.add_argument('--allow_duplicate', action='store_true')
    parser.add_argument('--no_parse_rel_label', action='store_true')

    args = parser.parse_args()
    main(args.openweb10th_path, args.ConcNet_Eng_path, args.out_path, args)
