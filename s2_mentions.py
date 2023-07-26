import pandas as pd
import os
from io import open
from conllu import parse_incr
from rapidfuzz import fuzz
from tqdm import tqdm
import pickle

def remove_elements_between_parentheses(sentence):
    opening = ['(', '[', '{']
    closing = [')', ']', '}']

    modified_sentence = []
    inside_parenthesis = False
    for element in sentence:
        if element['form'] in opening:
            inside_parenthesis = True
        if element['form'] in closing:
            inside_parenthesis = False
            continue
        if not inside_parenthesis:
            modified_sentence.append(element)

    return modified_sentence


tqdm.pandas()

parliaments = ['DK', 'FR', 'PL', 'RS', 'ES', 'UA']

for parliament in parliaments:
    parliament_path = '../data/parliaments/' + parliament + '/metasent_' + parliament + '.csv'
    parliament_path_unfiltered = '../data/parliaments/' + parliament + '/metadata_' + parliament + '.csv'

    df_parliament = pd.read_csv(parliament_path)

    df_parliament_unfiltered = pd.read_csv(parliament_path_unfiltered)

    members_all = df_parliament_unfiltered['Speaker_name'].unique()
    members_mps = df_parliament['Speaker_name'].unique()

    speech_ids = set(df_parliament['ID'].unique())
    speech_id_to_term = pd.Series(df_parliament.Term.values, index=df_parliament.ID).to_dict()

    speech_id_to_term = {k: speech_id_to_term[k] for k in speech_id_to_term if not pd.isna(speech_id_to_term[k])}

    term_to_speakers_all = df_parliament_unfiltered.groupby('Term')['Speaker_name'].unique().to_dict()
    term_to_speakers_mps = df_parliament.groupby('Term')['Speaker_name'].unique().to_dict()

    mentions_dictionary = {}

    conllu_files_path = '../data/parliaments/' + parliament + '/ParlaMint-' + parliament + '.conllu/'

    parliament_amgiuous_mentions = 0
    parliament_unambiguous_mentions_guests = 0
    parliament_unambiguous_mentions = 0

    for subdir, dirs, files in os.walk(conllu_files_path):
        for file in files:
            if 'conllu' in file and 'senat' not in file:
                file_path = os.path.join(subdir, file)
                conllu_file = open(file_path, "r", encoding="utf-8")

                process_speech = False
                speech_id = None
                mentions_set = set()
                for sentence in parse_incr(conllu_file):
                    if 'newdoc id' in sentence.metadata:
                        if len(mentions_set) > 0:
                            mentions_dictionary[speech_id] = mentions_set

                        mentions_set = set()
                        speech_id = sentence.metadata['newdoc id']

                        if speech_id in speech_ids:
                            process_speech = True

                    if process_speech:
                        # Finding the most probable person and adding an edge to the network
                        mention = None
                        sentence = remove_elements_between_parentheses(sentence)
                        for i in range(len(sentence)):
                            token = sentence[i]
                            if token['misc'] == None:
                                continue
                            NER = token['misc']['NER']
                            if NER == 'B-PER':
                                mention = token['lemma']
                            elif NER == 'I-PER' and mention != None:
                                mention = mention + ' ' + token['lemma']
                            else:
                                if mention != None and len(mention) > 2:

                                    max_similarity = 0
                                    match = None
                                    possible_people = []

                                    if speech_id in speech_id_to_term:
                                        term = speech_id_to_term[speech_id]
                                        members_all = term_to_speakers_all[term]
                                        members_mps = term_to_speakers_mps[term]
                                    else:
                                        members_all = df_parliament_unfiltered['Speaker_name'].unique()
                                        members_mps = df_parliament['Speaker_name'].unique()

                                    for possible_person in members_all:
                                        similarity = fuzz.token_set_ratio(mention, possible_person)
                                        if similarity == 100:
                                            possible_people.append(possible_person)

                                    if len(possible_people) == 1:
                                        match = possible_people[0]
                                        if match in members_mps:
                                            mentions_set.add(match)
                                            parliament_unambiguous_mentions += 1
                                        else:
                                            parliament_unambiguous_mentions_guests += 1

                                    elif len(possible_people) == 2 and (fuzz.token_set_ratio(possible_people[0],
                                                                                             possible_people[
                                                                                                 1]) == 100 or fuzz.token_set_ratio(
                                        possible_people[1],
                                        possible_people[0]) == 100):
                                        match = possible_people[0]
                                        if match in members_mps:
                                            mentions_set.add(match)
                                            parliament_unambiguous_mentions += 1
                                        else:
                                            parliament_unambiguous_mentions_guests += 1

                                    else:
                                        parliament_amgiuous_mentions += 1

                                    mention = None
                    else:
                        continue

                print("Done with file: ", file)

    mentions_path = '../data/parliaments/' + parliament + '/mentions_' + parliament + '_ver2.pickle'
    with open(mentions_path, 'wb') as handle:
        pickle.dump(mentions_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    print("Parliament stats: ", parliament)
    print("Parliament proper mentions: ", parliament_unambiguous_mentions)
    print("Parliament non-regular MPs mentions: ", parliament_unambiguous_mentions_guests)
    print("Parliament ambiguous mentions: ", parliament_amgiuous_mentions)

    print("Percentage of ambiguous mentions: ",
          round(parliament_amgiuous_mentions / (
                      parliament_unambiguous_mentions + parliament_amgiuous_mentions + parliament_unambiguous_mentions_guests),
                2))

    print('\n\n')
