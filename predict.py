from transformers import DistilBertForSequenceClassification, AutoTokenizer, pipeline
import torch
import shap

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and pipeline (only once)
model = DistilBertForSequenceClassification.from_pretrained("model_v4")
tokenizer = AutoTokenizer.from_pretrained("model_v4")
model.to(device).eval()

# Sentiment pipeline
sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  # instead of return_all_scores=True
    device=0 if torch.cuda.is_available() else -1
)

# SHAP Explainer
explainer = shap.Explainer(sentiment_pipeline)

# Label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to analyze text
def analyze_sentiment_with_shap(text: str):
    # Get SHAP values
    shap_values = explainer([text])
    
    # Predict sentiment
    pred_scores = sentiment_pipeline([text])[0]
    predicted_class = max(pred_scores, key=lambda x: x['score'])['label']
    predicted_index = int(predicted_class.split("_")[-1]) if "LABEL_" in predicted_class else int(predicted_class)
    predicted_sentiment = label_map[predicted_index]

    # SHAP text plot object (returns a matplotlib or HTML object depending on context)
    shap_plot = shap.plots.text(shap_values[0], display=False)  # display=False avoids inline rendering in notebooks

    return predicted_sentiment,shap_plot
