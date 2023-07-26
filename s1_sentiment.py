"""s1_sentiment.py: Sentiment prediction on parliamentary speeches"""

__author__ = "Bojan Evkoski"

import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pysbd
from tqdm import tqdm
import re


def remove_parentheses(text):
    # Remove text within parentheses
    text = re.sub(r'\([^()]*\)', '', text)

    # Remove text within square brackets
    text = re.sub(r'\[[^\[\]]*\]', '', text)

    # Remove text within curly braces
    text = re.sub(r'\{[^{}]*\}', '', text)

    return text


def remove_multispace(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text


tqdm.pandas()

parliaments = ['DK', 'FR', 'PL', 'RS', 'ES', 'UA']

seg = pysbd.Segmenter()

model_args = ClassificationArgs(early_stopping_patience=3, evaluate_each_epoch=True, num_train_epochs=20,
                                evaluate_during_training=True, train_batch_size=32, eval_batch_size=32,
                                regression=True, overwrite_output_dir=True, save_eval_checkpoints=False,
                                save_model_every_epoch=False, save_steps=-1, use_multiprocessing=False,
                                use_multiprocessing_for_evaluation=False, )

model = ClassificationModel(
    "xlmroberta", "../data/models/sentiment_model", use_cuda=True, args=model_args, num_labels=1,
)

for parliament in parliaments:
    parliament_path = '../data/parliaments/' + parliament + '/metadata_' + parliament + '.csv'

    df = pd.read_csv(parliament_path)
    print("Speeches before filtrations: ", len(df))

    df = df[df['Speaker_type'] == 'MP']
    df = df[df['Speaker_role'] == 'Regular']

    df['speech_clean'] = df['speech'].progress_apply(
        lambda x: remove_multispace(remove_parentheses(x)) if type(x) == str else x)

    df['speech_sentences'] = df['speech_clean'].progress_apply(lambda x: seg.segment(x) if type(x) == str else [])

    df = df[df['speech_sentences'].progress_apply(lambda x: len(x)) >= 5]
    print("Speeches after filtrations: ", len(df))

    df['speech_clean'] = df['speech_sentences'].progress_apply(lambda x: ' '.join(x[1:-1]))
    df['speech_clean'] = df['speech_clean'].apply(lambda x: " ".join(x.split()).strip())

    print("Preprocessing of parliament done. Starting sentiment...")

    save_path = '../data/parliaments/' + parliament + '/metasent_' + parliament + '.csv'

    df.to_csv(save_path, index=False)

    df = pd.read_csv(save_path)

    xlmr_predictions = model.predict(list(df['speech_clean'].values))
    df['sentiment'] = xlmr_predictions[0]
    df['sentiment'] = df['sentiment'].clip(-1, 1)

    df = df.drop(columns=['speech_sentences'])

    df.to_csv('../data/parliaments/' + parliament + '/metasent_' + parliament + '.csv', index=False)
