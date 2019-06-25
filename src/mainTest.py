from src.GetData import downloadSpeechData, getDataDict

if __name__ == "__main__":
    # Download dataset
    downloadSpeechData(data_path='../speechData/')
    
    # Get dict containing path and labels 
    dataDict = getDataDict(data_path='../speechData/')

    # Perform exploratory analysis
    print(len(dataDict['train']['labels']))