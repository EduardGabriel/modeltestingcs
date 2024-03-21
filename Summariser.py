from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class Summariser():

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.pipeline = self.setup_pipe()
        except ValueError as e:
                if "summarization" in str(e):
                    print(f"Warning: {model_name} might not be suitable for summarization. Consider using a different model. Error: {e}")
                else:
                     raise e


    def setup_pipe(self):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)  # Load the model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Load the tokenizer
            self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            return self.pipeline
    
    def summarize(self, text, min_length=10):
        
        """
        Summarises the given text using the loaded model

        Returns:
            str: The summarized text or an informative message if summarization fails.

        """

        # Get tokenized input length 
        input_tokens = self.tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        input_length = len(input_tokens[0])
        max_length = int(input_length//1.8)

        try:
            summary = self.pipeline(text, max_length = max_length, min_length = min_length)
            return summary[0].get("summary_text", "An error occured while summarizing the text.")
        
        except Exception as e:
             
             print(f"Error during summarization: {e}")
             return "An error occurred while summarizing the text."
        



