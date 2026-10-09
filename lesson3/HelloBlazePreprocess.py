"""Preprocessing script for the HelloBlaze BlazingText training workflow.

This module is written to be executed as a SageMaker Processing job entry
point (see lesson 2, exercise 4). It unzips the Amazon reviews archive,
labels each review as helpful or unhelpful from its vote counts, splits
reviews into individual sentences, and writes 90/10 train/test splits to
the processing output directories.
"""

import json
import zipfile


def unzip_data(input_data_path):
    """Extract a zipped reviews archive into the current directory.

    Args:
        input_data_path: Path to the .zip archive holding the reviews file.

    Returns:
        The file name of the first entry inside the archive.
    """
    with zipfile.ZipFile(input_data_path, 'r') as input_data_zip:
        input_data_zip.extractall('.')
        return input_data_zip.namelist()[0]


# Input data is a file with a single JSON object per line with the following format:
# {
#  "reviewerID": <string>,
#  "asin": <string>,
#  "reviewerName" <string>,
#  "helpful": [
#    <int>, (indicating number of "helpful votes")
#    <int>  (indicating total number of votes)
#  ],
#  "reviewText": "<string>",
#  "overall": <int>,
#  "summary": "<string>",
#  "unixReviewTime": <int>,
#  "reviewTime": "<string>"
# }
#
# We are specifically interested in the fields "helpful" and "reviewText"


def label_data(input_data):
    """Label reviews as helpful or unhelpful based on their vote counts.

    A review whose helpful votes are a strict majority of total votes is
    labeled "__label__1"; one whose helpful votes are a strict minority is
    labeled "__label__2". Reviews with no votes, or an exact 50/50 split,
    are dropped.

    Args:
        input_data: Path to the JSON-lines reviews file.

    Returns:
        A list of strings, each starting with the label followed by the
        review text.
    """
    labeled_data = []
    HELPFUL_LABEL = "__label__1"
    UNHELPFUL_LABEL = "__label__2"

    with open(input_data, 'r') as reviews_file:
        for l in reviews_file:
            l_object = json.loads(l)
            helpful_votes = float(l_object['helpful'][0])
            total_votes = l_object['helpful'][1]
            reviewText = l_object['reviewText']
            if total_votes != 0:
                if helpful_votes / total_votes > .5:
                    labeled_data.append(" ".join([HELPFUL_LABEL, reviewText]))
                elif helpful_votes / total_votes < .5:
                    labeled_data.append(" ".join([UNHELPFUL_LABEL, reviewText]))

    return labeled_data


# Labeled data is a list of sentences, starting with the label defined in label_data.


def split_sentences(labeled_data):
    """Split each labeled review into sentences, keeping the review's label.

    BlazingText trains best on many short sentences rather than a few long
    paragraphs, so reviews are split on "." and every non-empty sentence
    keeps the original review label.

    Args:
        labeled_data: List of "label review-text" strings from label_data.

    Returns:
        A list of "label sentence" strings.
    """
    new_split_sentences = []
    for d in labeled_data:
        label = d.split()[0]
        sentences = " ".join(d.split()[1:]).split(".")  # Split off the label, then split sentences
        for s in sentences:
            if s:  # Skip empty sentences. Common with "..."
                new_split_sentences.append(" ".join([label, s]))
    return new_split_sentences


def write_data(data, train_path, test_path, proportion):
    """Write labeled data to train/test files split at the given proportion.

    Args:
        data: List of labeled data lines.
        train_path: Destination path for the training split.
        test_path: Destination path for the test split.
        proportion: Fraction of lines written to the training file.
    """
    border_index = int(proportion * len(data))
    with open(train_path, 'w') as train_f, open(test_path, 'w') as test_f:
        for index, d in enumerate(data):
            if index < border_index:
                train_f.write(d + '\n')
            else:
                test_f.write(d + '\n')


if __name__ == "__main__":
    unzipped_path = unzip_data('/opt/ml/processing/input/reviews_Musical_Instruments_5.json.zip')
    labeled_data = label_data(unzipped_path)
    new_split_sentence_data = split_sentences(labeled_data)
    write_data(new_split_sentence_data,
               '/opt/ml/processing/output/train/hello_blaze_train_scikit',
               '/opt/ml/processing/output/test/hello_blaze_test_scikit',
               .9)
