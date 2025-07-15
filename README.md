#  Amazon Product Review Sentiment Analysis using DistilBERT

A lightweight, interactive web app that performs sentiment analysis on Amazon product reviews using a fine-tuned DistilBERT model. Built with <strong>Streamlit</strong>, this tool offers fast predictions and SHAP-based interpretability to visualize which words influenced the sentiment.

---

##  Project Overview

<ul>
  <li>Analyzes sentiment (Positive / Negative / Neutral) of Amazon reviews</li>
  <li>Built using DistilBERT transformer model from HuggingFace</li>
  <li>Interactive UI with real-time predictions and SHAP visualizations</li>
  <li>Handles class imbalance using upsampling techniques</li>
</ul>

---

##  Features

<ul>
  <li> Real-time sentiment prediction</li>
  <li> SHAP-based word importance visualizations</li>
  <li> Upsampling to address class imbalance</li>
  <li> Input box to test custom reviews</li>
  <li> Word cloud for dataset insights</li>
  <li> Model performance and metric plots</li>
</ul>

---

##  Tools and Tech Stack Used

###  Frontend / UI

<ul>
  <li><strong>Streamlit</strong> — Interactive web dashboard and UI components</li>
  <li><strong>streamlit-shap</strong> — Embed SHAP plots into the Streamlit app</li>
</ul>

###  Backend / Model

<ul>
  <li><strong>Python</strong> — Main language for backend and ML logic</li>
  <li><strong>transformers (HuggingFace)</strong> — DistilBERT for sentiment classification</li>
  <li><strong>PyTorch</strong> — Loads model and runs inference</li>
  <li><strong>SHAP</strong> — Model interpretability and explainability</li>
</ul>

###  Data Visualization & Utilities

<ul>
  <li><strong>Matplotlib</strong> — Visual plots and metrics</li>
  <li><strong>WordCloud</strong> — Generate word cloud visuals</li>
  <li><strong>pandas, numpy</strong> — Data preprocessing and handling</li>
  <li><strong>seaborn</strong> — Optional advanced visualizations</li>
  <li><strong>Streamlit Session State</strong> — Manage session, inputs, navigation</li>
</ul>

---

##  System Workflow

<pre>
User Review → Tokenization → DistilBERT Prediction → SHAP Explanation → UI Output
</pre>

---

##  Model Information

<ul>
  <li>Model: <code>distilbert-base-uncased-finetuned-sst-2-english</code> (fine-tuned)</li>
  <li>Training: Includes <strong>upsampling</strong> for class balancing</li>
  <li>Interpretability: SHAP highlights word influence on prediction</li>
</ul>

---

##  Contributors

<ul>
  <li><a href="https://github.com/Uthra-karunakaran" target="_blank">Uthra</a></li>
  <li><a href="https://github.com/Web-Dev-Learner" target="_blank">Shrinedhi M R</a></li>
</ul>


---


##  Key Highlights

<ul>
  <li> Fast, lightweight DistilBERT model</li>
  <li> Word-level SHAP explanations for transparency</li>
  <li> Class balancing improves accuracy on skewed data</li>
  <li> Deploy-ready Streamlit web interface</li>
</ul>
