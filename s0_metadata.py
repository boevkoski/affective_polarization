"""s0_metadata.py: Reading ParlaMint .conllu files and outputing .csvs with  speech metadata"""

__author__ = "Bojan Evkoski"

import pandas as pd
import os
from conllu import parse_incr


def metadata_to_csv(directory):
    df = pd.DataFrame()
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if 'meta.tsv' in file and 'senat' not in file:
                file_path = os.path.join(subdir, file)
                df_meta = pd.read_csv(file_path, sep='\t')
                df = pd.concat([df, df_meta])

    return df


def speech_to_csv(directory, df):
    ID_to_text = {}

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if 'conllu' in file and 'senat' not in file:
                file_path = os.path.join(subdir, file)
                conllu_file = open(file_path, "r", encoding="utf-8")

                speech = ""
                speech_id = None
                for sentence in parse_incr(conllu_file):
                    if 'newdoc id' in sentence.metadata:
                        if speech_id != None:
                            ID_to_text[speech_id] = speech.rstrip()
                        speech_id = sentence.metadata['newdoc id']
                        speech = ""

                    speech += sentence.metadata['text'] + ' '

                ID_to_text[speech_id] = speech.rstrip()
                print("Done with file: ", file)

    df['speech'] = df.ID.apply(lambda x: ID_to_text.get(x, None))

    return df


parliaments = ['DK', 'FR', 'PL', 'RS', 'ES', 'UA']

for parliament in parliaments:
    parliament_folder = '../data/parliaments/' + parliament + '/ParlaMint-' + parliament + '.conllu'

    df = metadata_to_csv(parliament_folder)

    print("Done with metadata: ", parliament)

    df = speech_to_csv(parliament_folder, df)

    print("Done with speeches: ", parliament)

    df.to_csv('../data/parliaments/' + parliament + '/metadata_' + parliament + '.csv', index=False)

    print("Done with parliament: ", parliament)
