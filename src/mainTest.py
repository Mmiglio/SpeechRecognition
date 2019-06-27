from getData import downloadSpeechData, getDataDict
from constants import categories, inv_categories
import pandas as pd

if __name__ == "__main__":
    # Download dataset
    downloadSpeechData(data_path='../speechData/')

    # Get dict containing path and labels
    dataDict = getDataDict(data_path='../speechData/')

    # Operations on data
    trainDF = pd.DataFrame(dataDict['train'])
    valDF = pd.DataFrame(dataDict['val'])
    testDF = pd.DataFrame(dataDict['test'])

    DFlist = [trainDF, valDF, testDF]

    # Add category label and drop file containing background noise
    # (they need further processing)
    for df in DFlist:
        df['category'] = df.apply(
            lambda row: inv_categories[row['labels']], axis=1
            )
        df = df.loc[df['category'] != '_background_noise_', :]

    print("Train DataFrame: {}".format(trainDF.shape[0]))
    print("Validation DataFrame: {}".format(valDF.shape[0]))
    print("Test DataFrame: {}".format(testDF.shape[0]))
