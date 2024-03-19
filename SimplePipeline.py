from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

### TODO: 
### GENERALLY HANDLE tokenization, passing through model, post processing
### Methods for loading datasets
### Methods for finetuning 
### Evaluation of models


class Runner:
    def __init__(self):
        self.model = None
        self.model_name = None
    
    def setup_model(self, model_name):
        self.model = pipeline("summarization", model=model_name)
    
    def run_model(self, documents):
        summary = self.model(documents)
        return summary

class TextSummariser:
    def __init__(self, model_file):
        self.models = self.load_models(model_file)

    def load_models(self, model_file):
        with open(model_file, 'r') as file:
            return [line.strip() for line in file]

    def run_summarization(self, text):
        for model in self.models:
            runner = Runner()
            runner.setup_model(model)
            summary = runner.run_model(text)
            print("Model:", model, "\n", summary)




def main():
    test_set = """America has changed dramatically during recent years. Not only has the number of graduates in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering declined, but in most of the premier American universities engineering curricula now concentrate on and encourage largely the study of engineering science. As a result, there are declining offerings in engineering subjects dealing with infrastructure, the environment, and related issues, and greater concentration on high technology subjects, largely supporting increasingly complex scientific developments. While the latter is important, it should not be at the expense of more traditional engineering.
    Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance the teaching of engineering. Both China and India, respectively, graduate six and eight times as many traditional engineers as does the United States. Other industrial countries at minimum maintain their output, while America suffers an increasingly serious decline in the number of engineering graduates and a lack of well-educated engineers.
    """
    summarizer = TextSummariser("models.txt")
    summarizer.run_summarization(test_set)


if __name__ == "__main__":
    main()