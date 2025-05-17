from streamlit_shap import st_shap
import streamlit as st
from predict import analyze_sentiment_with_shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Streamlit configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Session state to store navigation and feedback status

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "liked" not in st.session_state:
    st.session_state.liked = False
if "disliked" not in st.session_state:
    st.session_state.disliked = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# Sidebar Navigation 
st.sidebar.title("📊 Navigation")
choice = st.sidebar.radio("Select an option", ["Home", "Sentiment Analysis", "Metrics & Visualization", "Feedback & Support"], key="unique_radio_key")



# Function: Feedback & Support Section
def show_feedback_and_support():
    st.title("📬 Feedback & Support")

    # Section 1: Feedback on Prediction
    st.markdown("---")
    st.subheader("📝 Was this prediction helpful?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Like", key="like_button"):
            st.session_state.liked = True
            st.session_state.disliked = False
    with col2:
        if st.button("👎 Dislike", key="dislike_button"):
            st.session_state.disliked = True
            st.session_state.liked = False

    comment = st.text_area("💬 Leave a comment (optional)", placeholder="Tell us what you think...")

    if st.button("✅ Submit Feedback", key="submit_button"):
        if st.session_state.liked:
            st.success("🎉 Thanks for the like! Your feedback is appreciated.")
            st.session_state.feedback_submitted = True
        elif st.session_state.disliked:
            st.info("🧐 Thanks for your feedback! We'll use it to improve.")
            st.session_state.feedback_submitted = True
        else:
            st.warning("⚠️ Please select Like or Dislike before submitting.")

        if comment.strip():
            st.markdown("🗒️ **Your comment:**")
            st.write(comment)






    # Section 2: Report a Problem
    st.markdown("---")
    st.subheader("🐞 Found an issue?")
    issue_title = st.text_input("Issue Title", placeholder="e.g., App crashes after prediction")
    issue_description = st.text_area("Describe the issue", placeholder="Provide steps to reproduce or more details...")
    uploaded_image = st.file_uploader("📷 Upload a screenshot (optional)", type=["png", "jpg", "jpeg"])

    if st.button("🚨 Submit Issue Report"):
        if issue_title.strip() and issue_description.strip():
            st.success("✅ Your issue has been submitted. Thank you!")
            
        else:
            st.error("⚠️ Please fill in both the issue title and description.")

    # Section 3: Contact Developer
    st.markdown("---")
    st.subheader("👩‍💻 Need to get in touch?")
    
    st.write("Feel free to connect with the developers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            [![LinkedIn](https://img.shields.io/badge/-Shrinedhi%20M%20R-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shrinedhi-m-r)
            [![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github)](https://github.com/Web-Dev-Learner)
            """
        )

    with col2:
        st.markdown(
            """
            [![LinkedIn](https://img.shields.io/badge/-Uthra%20Karuna-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/uthra-karuna/)
            [![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github)](https://github.com/Uthra-karunakaran)
            """
        )




# # Label mapping M 2 SHAP

# label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# # Load model, tokenizer, pipeline
# @st.cache_resource
# def load_model_pipeline():
#     device = torch.device("cpu")
#     model = DistilBertForSequenceClassification.from_pretrained("model_v4")
#     tokenizer = AutoTokenizer.from_pretrained("model_v4")
#     model.to(device)
#     model.eval()
#     sentiment_pipeline = pipeline(
#         "text-classification",
#         model=model,
#         tokenizer=tokenizer,
#         return_all_scores=True,
#         device=-1  # CPU
#     )
#     return model, tokenizer, sentiment_pipeline

# # Load SHAP explainer (use _ to avoid unhashable errors)
# @st.cache_resource
# def get_explainer(_sentiment_pipeline):
#     return shap.Explainer(_sentiment_pipeline)

# # Analysis function
# def analyze_sentiment_with_shap(text):
#     model, tokenizer, sentiment_pipeline = load_model_pipeline()
#     explainer = get_explainer(sentiment_pipeline)
#     shap_values = explainer([text])
#     scores = sentiment_pipeline([text])[0]
#     predicted_class = max(scores, key=lambda x: x['score'])['label']
#     predicted_index = int(predicted_class.split("_")[-1]) if "LABEL_" in predicted_class else int(predicted_class)
#     sentiment = label_map[predicted_index]
#     return sentiment, shap_values



# Home Page
if choice == "Home":

    st.title("🛍️ Amazon Product Review Sentiment Analysis Dashboard")
    st.markdown("---")

    # Overview / Introduction
    st.subheader("📌 Overview")
    st.markdown("""
    Welcome to the **Amazon Product Review Sentiment Analysis Dashboard** – a powerful NLP-based tool designed to analyze and explain the sentiment behind customer reviews.

    Whether you're a business analyzing feedback or a curious user exploring sentiment trends, this dashboard empowers you with AI insights, interactive visuals, and explainability using SHAP values.
    """)

    # How to Use / User Guide
    st.subheader("📖 How to Use")
    st.markdown("""
    1. Navigate to the **Sentiment Analysis** section from the sidebar.
    2. Enter or paste any Amazon product review into the input box.
    3. Click on **Analyze** to get the predicted sentiment (Positive/Negative/Neutral).
    4. Explore **SHAP visualizations** to understand which words impacted the prediction.
    5. View a **Word Cloud** of your input text.
    6. Submit feedback or suggestions in the **Review** section.
    """)

    # Key Features
    st.subheader("🚀 Key Features")
    st.markdown("""
    - ✅ Sentiment Classification using **DistilBERT**
    - 🧠 Explainable AI with **SHAP (SHapley Additive exPlanations)**
    - ☁️ Visual word cloud generation
    - 📊 Clean and interactive UI with **Streamlit**
    - 💬 Feedback collection system
    """)

    # Tech Stack
    st.subheader("🛠️ Tech Stack Used")
    st.markdown("""
    - **Frontend**: Streamlit  
    - **Backend / NLP**: Python, HuggingFace Transformers (DistilBERT)  
    - **Explainability**: SHAP  
    - **Visualization**: Matplotlib, WordCloud  
    - **Deployment-ready**: Easily deployable on Streamlit Cloud
    """)

    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - Try using both positive and negative reviews to test accuracy.
    - Use real Amazon review examples for best results.
    - Avoid empty inputs or emoji-only content.
    """)

   

# Sentiment Analysis Page

elif choice == "Sentiment Analysis":


    st.session_state.page = "Sentiment Analysis"
    st.title("🧠 Sentiment Analysis")
    user_input = st.text_area("📝 Enter your product review")

    if st.button("Analyze", key="analyze"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a review!")
        else:
            sentiment, shap_plot = analyze_sentiment_with_shap(user_input)
            st.success(f"✅ Predicted Sentiment: **{sentiment.upper()}**")

            st.subheader("🔍 Word-level Impact")
            st_shap(shap_plot)

            st.subheader("☁️ Word Cloud from Review")
            wordcloud = WordCloud(background_color='white').generate(user_input)
            fig2, ax2 = plt.subplots()
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig2)




# # ---- STREAMLIT UI ----  M - 2

# elif choice == "Sentiment Analysis":
#     st.session_state.page = "Sentiment Analysis"
#     st.title("🧠 Sentiment Analysis")
#     user_input = st.text_area("📝 Enter your product review")

#     if st.button("Analyze", key="analyze"):
#         if not user_input.strip():
#             st.warning("⚠️ Please enter a review!")
#         else:
#             with st.spinner("Analyzing..."):
#                 sentiment, shap_values = analyze_sentiment_with_shap(user_input)
#                 st.success(f"✅ Predicted Sentiment: **{sentiment}**")

#                 # SHAP interactive plot in Streamlit
#                 st.subheader("🔍 SHAP Explanation")
#                 st_shap(shap.plots.text(shap_values[0]), height=400)




# visualization




elif choice == "Metrics & Visualization":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import streamlit as st

    st.session_state.page = "Metrics & Visualization"
    st.title("📈 Evaluation Metrics & Visualization")

    # Overall metrics
    precision = 0.8032
    recall = 0.7984
    f1 = 0.7989

    st.subheader("✅ Overall Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")
    col3.metric("F1-Score", f"{f1:.4f}")

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Classification Report Data")

    # Classification report data
    data = {
        "Label": ["Class 0", "Class 1", "Class 2", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.8348, 0.6947, 0.8798, None, 0.8031, 0.8032],
        "Recall": [0.7280, 0.7699, 0.8968, 0.7984, 0.7982, 0.7984],
        "F1-score": [0.7777, 0.7304, 0.8882, None, 0.7988, 0.7989],
        "Support": [28935, 28985, 29149, 87069, 87069, 87069]
    }
    df_report = pd.DataFrame(data)
    st.dataframe(df_report.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-score": "{:.4f}",
        "Support": "{:,.0f}"
    }))

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📉 Overall Metrics Bar Chart")

    # 1. Plot Overall Metrics bar chart
    overall_metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
    fig1, ax1 = plt.subplots()
    bars = ax1.bar(overall_metrics.keys(), overall_metrics.values(), color=['#4daf4a', '#377eb8', '#ff7f00'])
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall Evaluation Metrics')
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')
    st.pyplot(fig1)

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Per-Class Precision, Recall & F1-Score")

    # Per-class scores for plotting
    classes = ['Class 0', 'Class 1', 'Class 2']
    precision_scores = [0.8348, 0.6947, 0.8798]
    recall_scores = [0.7280, 0.7699, 0.8968]
    f1_scores = [0.7777, 0.7304, 0.8882]
    support_counts = [28935, 28985, 29149]

    # 2. Plot per-class grouped bar chart
    x = np.arange(len(classes))  # label locations
    width = 0.25

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    rects1 = ax2.bar(x - width, precision_scores, width, label='Precision', color='#4daf4a')
    rects2 = ax2.bar(x, recall_scores, width, label='Recall', color='#377eb8')
    rects3 = ax2.bar(x + width, f1_scores, width, label='F1-Score', color='#ff7f00')

    ax2.set_ylabel('Scores')
    ax2.set_title('Evaluation Metrics by Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1)
    ax2.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.3f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    st.pyplot(fig2)

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🥧 Support (Sample Count) Distribution by Class")

    # 3. Pie chart for Support (sample distribution per class)
    fig3, ax3 = plt.subplots()
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    ax3.pie(support_counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=colors)
    ax3.axis('equal')  # Equal aspect ratio ensures pie is a circle.
    ax3.set_title('Support (Sample Distribution) per Class')
    st.pyplot(fig3)

    # Final note
    st.markdown("<br>", unsafe_allow_html=True)
    


# Review Page
elif choice == "Feedback & Support":

    st.session_state.page = "Feedback & Support"
    show_feedback_and_support()



