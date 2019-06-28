import os
import requests
import tarfile
import pandas as pd

# Dictionary containing categories + encoded value
from constants import categories, inv_categories


def downloadSpeechData(data_path='speechData/'):
    """
    Download google speech commands dataset
    """
    data_path = os.path.abspath(data_path)+'/'
    datasets = ['train', 'test']
    urls = [
        'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        'http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz'
    ]

    for dataset, url in zip(datasets, urls):
        dataset_directory = data_path + dataset
        # Check if we need to extract the dataset
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
            file_name = data_path + dataset + '.tar.gz'
            # Check if the dataset has been downloaded
            if os.path.isfile(file_name):
                print('{} already exists. Skipping download.'.format(file_name))
            else:
                downloadFile(url=url, file_name=file_name)

            # extract downloaded file
            extractFile(file_name=file_name, directory=dataset_directory)
        else:
            print('Nothing to do.')


def getDataDict(data_path='speechData/'):
    """
    Get the list of .Wav files and relative label
    """
    valWavs = open(data_path + 'train/validation_list.txt').read().splitlines()
    testWavs = open(data_path + 'train/testing_list.txt').read().splitlines()

    valWavs = ['speechData/train/'+f for f in valWavs]
    testWavs = ['speechData/train/'+f for f in testWavs]

    # Find trainWavs as allFiles / {testWavs, valWavs}
    allFiles = list()
    for root, dirs, files in os.walk(data_path+'train/'):
        allFiles += [root+'/' + f for f in files if f.endswith('.wav')]
    trainWavs = list(set(allFiles)-set(valWavs)-set(testWavs))

    # Final evaluation set
    finalTestWavs = list()
    for root, dirs, files in os.walk(data_path+'test/'):
        finalTestWavs += [root+'/' + f for f in files if f.endswith('.wav')]

    # Get labels
    valWavLabels = [getLabel(wav) for wav in valWavs]
    testWavLabels = [getLabel(wav) for wav in testWavs]
    trainWavLabels = [getLabel(wav) for wav in trainWavs]
    finalTestWavLabels = [getLabel(wav) for wav in finalTestWavs]

    # Create dictionaries containinf (file, labels)
    trainData = {'files': trainWavs, 'labels': trainWavLabels}
    valData = {'files': valWavs, 'labels': valWavLabels}
    testData = {'files': testWavs, 'labels': testWavLabels}
    finalTestData = {'files': finalTestWavs, 'labels': finalTestWavLabels}

    dataDict = {
        'train': trainData,
        'val': valData,
        'test': testData,
        'finalTest': finalTestData
        }

    return dataDict


def downloadFile(url, file_name):
    """
    Download a file.
    """
    data_request = requests.get(url)
    print('Downloading {} into {}'.format(url, file_name))
    with open(file_name, 'wb') as f:
        f.write(data_request.content)


def extractFile(file_name, directory):
    """
    Extract dataset
    """
    print('Extracting {} into {}'.format(file_name, directory))
    if (file_name.endswith("tar.gz")):
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=directory)
        tar.close()
    else:
        print('Unknown format.')


def getLabel(file_name):
    """
    Get the label from file path
    path = */baseDir/train/CATEGORY/file_name
    """
    category = file_name.split('/')[-2]
    return categories.get(category, categories['_background_noise_'])


def getDataframe(data):
    """
    Create a dataframe from a Dictionary and remove _background_noise_
    """
    df = pd.DataFrame(data)
    df['category'] = df.apply(
        lambda row: inv_categories[row['labels']], axis=1
        )
    df = df.loc[df['category'] != '_background_noise_', :]

    return df
