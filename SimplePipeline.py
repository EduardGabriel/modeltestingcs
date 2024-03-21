from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from Summariser import Summariser
import baseline_summarizer


 
class TextSummariser:
    def __init__(self, model_list):
        self.model_list = model_list

    

    def run_summarization(self, text_data):
        for i, text in enumerate(text_data):
            print(f"\n** Text #{i+1} **\n", text)  # Clear separator for each text entry
            baseline_summary = baseline_summarizer.text_rank_summarize(text, 0.4)
            print(f"\n** Baseline summary: {(baseline_summary)}")
            for model_name in self.model_list:
                summariser = Summariser(model_name)  #Create  a summariser object object, instantiating a pipeline 
                summary = summariser.summarize(text) #Generate summary
                
                print(f"\n** Model: {model_name} **\n")
                print(f"**Summary: {summary}\n**")



survey_data = [
    "What did you like most about your recent experience with our company? The customer service was exceptional. The representative was very helpful and knowledgeable.",
    "What features of our product do you find most useful? The reporting functionality is very helpful for analyzing data. I appreciate the ease of use and the intuitive interface. The integration with other applications saves me a lot of time.",
    "Do you have any suggestions for how we can improve our product? It would be great to have a mobile app for on-the-go access. Adding more customization options for reports would be beneficial. Including tutorials or explainer videos for new users would be helpful.",
]

model_list = ["Falconsai/text_summarization", "philschmid/bart-large-cnn-samsum", "philschmid/distilbart-cnn-12-6-samsum"]


def main():
   
  



    summarizer = TextSummariser(model_list)
    summarizer.run_summarization(survey_data)
    

if __name__ == "__main__":
    main()