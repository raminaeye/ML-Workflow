"""Lambda-friendly preprocessing for the HelloBlaze BlazingText workflow.

Same preprocessing pipeline as HelloBlazePreprocess.py (unzip, label reviews
from vote counts, split into sentences, write train/test splits), adapted for
AWS Lambda: the input archive is downloaded from S3 into /tmp, the outputs
are written to /tmp, and both are uploaded back to S3.

Before deploying, set BUCKET_NAME and PREFIX below to the S3 bucket and key
prefix you want the processed data written to.
"""

import json
import logging
import zipfile
import os
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = 'FILL_THIS_IN'
PREFIX = 'FILL_THIS_IN'


def unzip_data(input_data_path):
    """Extract a zipped reviews archive into /tmp.

    Args:
        input_data_path: Path to the .zip archive holding the reviews file.

    Returns:
        The /tmp path of the first entry inside the archive.
    """
    with zipfile.ZipFile(input_data_path, 'r') as input_data_zip:
        input_data_zip.extractall('/tmp/')
        return '/tmp/' + input_data_zip.namelist()[0]


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


def upload_data(file_name):
    """Upload a local file to S3 under PREFIX in BUCKET_NAME.

    Args:
        file_name: Local path of the file to upload.

    Returns:
        False if the upload fails, None otherwise.
    """
    object_name = os.path.join(PREFIX, os.path.basename(file_name))
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, BUCKET_NAME, object_name)
    except ClientError as e:
        logging.error(e)
        return False


def write_data(data, b_name, proportion):
    """Write train/test splits to /tmp and upload both to S3.

    Args:
        data: List of labeled data lines.
        b_name: Base name used to derive the train/test file names.
        proportion: Fraction of lines written to the training file.
    """
    train_path = '/tmp/' + b_name + '_train'
    test_path = '/tmp/' + b_name + '_test'
    border_index = int(proportion * len(data))
    with open(train_path, 'w') as train_f, open(test_path, 'w') as test_f:
        for index, d in enumerate(data):
            if index < border_index:
                train_f.write(d + '\n')
            else:
                test_f.write(d + '\n')
    upload_data(train_path)
    upload_data(test_path)


def download_data(s3_input_uri):
    """Download an S3 object to /tmp.

    Args:
        s3_input_uri: S3 URI of the form "bucket/key" (no s3:// scheme).

    Returns:
        The /tmp path of the downloaded file.
    """
    s3 = boto3.client('s3')
    input_bucket = s3_input_uri.split('/')[0]
    input_object = '/'.join(s3_input_uri.split('/')[1:])
    file_name = '/tmp/' + os.path.basename(input_object)
    s3.download_file(input_bucket, input_object, file_name)
    return file_name


def preprocess(s3_input_uri):
    """Run the full preprocessing pipeline on a zipped reviews archive in S3.

    Downloads the archive, unzips it, labels the reviews, splits them into
    sentences, and writes 90/10 train/test splits back to S3.

    Args:
        s3_input_uri: S3 URI of the zipped reviews archive, of the form
            "bucket/key" (no s3:// scheme).
    """
    f_name = download_data(s3_input_uri)
    unzipped_path = unzip_data(f_name)
    labeled_data = label_data(unzipped_path)
    new_split_sentence_data = split_sentences(labeled_data)
    write_data(new_split_sentence_data, os.path.basename(s3_input_uri), .9)
