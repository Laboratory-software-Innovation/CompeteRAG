## Dependencies 

This is a critical part, because you need to ensure that the libraries are of correct version so that venv has everything needed to run. Before running the API, please install the requirements from req.txt.

```
pip3 install -r req.txt
```

The above requirements file should include all of the needed libraries. However, in case if you run in to issues with Kaggle, please run the its library installation with sudo.

```
sudo pip3 install kaggle
```

## Selenium Helper

The repository also contains two version of selenium helper functions, for Firefox and Edge browsers. I prefer using the Firefox one, however, you are free to switch it for your own too (e.g Chrome). Whichever one you pick, make sure you rename it to selenium_helper.py

## ChatGPT - API Key
Next, dont forget to add the API key to the directory
1. Create a .env file
2. Add this 
```
OPENAI_API_KEY = sk-...
```

## Kaggle Configuration
Finally, don't forget to add your .kaggle configuration file. 

For Ubuntu:
```
~/config/kaggle/.kaggle
```

The directory may vary depending on the system. 

## Kaggle Competitions

Before you start parsing the competitions, ensure that you have joined the competitions you are aiming to parse, otherwise you will run into the HTTP Error and the API will not be able to retrieve the test.csv dataset file. 

```
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://www.kaggle.com/api/v1/competitions/data/download/slug/train.csv
```


## Collecting and Encoded Matrix From the Notebooks Metadata

Before we can even compare to any notebooks, we first need to get at least a small dataset. Therefore, we need to collect competitions, their metadata, its datasets and solutions. First, it parses each competition's html page and strips it to text. Next, downloads the dataset (train.csv file), runs it through a summarization function and provides a dense summary of the dataset. It is then sent to an LLM to provide as a much more dense but still helpful summary. It then filters the notebooks section of every single competition by TensorFlow and PyTorch. The resulting notebooks are then sent through an LLM for more deep analysis, for example if the LLM understands that the notebook uses a Machine Learning algorithms, it skips that notebooks and moves on. 

Finally as soon as we have all the notebooks metadata saved and structured we build an encoded matrix for the later comparison.

```
python3 rag.py cb
```

## Building the Code for a New Competition

The final step to find the notebooks solutions that are as close as possible to our new competitions. Now, the API will find top-k solutions similar to the provided competitions, and append them to our next LLM prompt. 

```
python3 rag.py code <top-k> <simplified 0|1 only for keras tuner> <keras-tuner 0|1>
```

<top-k> - top-k solution you would like to feed into the LLM for relying

<simplified> - boolean, if 1 picks an exact hyperparameter blueprint for Keras Tuner, 0 - loads up all blueprints for the LLM to decide on its own

<keras-tuner> - boolean, only works if the keras code already exists. If 1 takes the existing Keras code and turns it into its Keras Tuner version

## Results

The API should then generate two files: 

* .json description of a competition
* .py model code (you may need to remove \<Code> and \</Code> tags from top and bottom respectively).

## Follow-up Prompt

If, in any case the initial prompt generated a faulty code, and you receive an error message, copy the error message and insert it below the code segment into a new \<Error>...\</Error> block segment

# P.S.

The RAG does seem to do most of the tasks with only one attempt and sometimes the score is 10% - 15% better. However, the bigger problem is to ensure it does consistently and follow the correct TF Keras code.
