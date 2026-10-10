import io
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import streamlit as st

# === BLOCK 1: Load and Clean Core Data ===
df = pd.read_csv("mental_health_combined_test.csv")
df = df.dropna(subset=["text", "status"])

# === BLOCK 2: Feature Extraction ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["status"]

# 🛠️ ML SPLITTING: 80% Training, 20% Testing
# We extract the raw sentences here as well so we can read them during error analysis
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, df["text"], test_size=0.2, random_state=42
)

# === BLOCK 3: Model Training ===
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# === BLOCK 4: UI Dashboard Layout ===
st.set_page_config(page_title="SentX Dashboard", layout="wide")
st.title("🛡️ SentX: Advanced Behavioral Risk Dashboard")

color_map = {
    "Normal": "#2ecc71",
    "Anxiety": "#3498db",
    "Depression": "#f1c40f",
    "Suicidal": "#e74c3c",
}

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Text Screening",
    "📂 Bulk CSV Batch Processing",
    "🔬 Computational Model Diagnostics"
])

# --- TAB 1: SINGLE TEXT MODE ---
with tab1:
    st.subheader("Individual Entry Analysis")
    user_text = st.text_area(
        "Paste anonymous message logs, journal entries, or forum text below:",
        "i don't understand whats wrong with me. i don't know why i freak out sometimes and feel trapped and panicked.",
        key="single_input",
    )

    if st.button("Execute Single Screening"):
        user_data = vectorizer.transform([user_text])
        probabilities = model_lr.predict_proba(user_data)[0]
        classes = model_lr.classes_

        chart_data = pd.DataFrame(
            {"Psychological Vector": classes, "Probability (%)": probabilities * 100}
        )
        chart_data = chart_data.sort_values(by="Probability (%)", ascending=True)

        fig = px.bar(
            chart_data,
            x="Probability (%)",
            y="Psychological Vector",
            orientation="h",
            color="Psychological Vector",
            color_discrete_map=color_map,
            text="Probability (%)",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis=dict(range=[0, 100]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
        )
        st.plotly_chart(fig, width="stretch")

        prob_dict = dict(zip(classes, probabilities * 100))
        predicted_state = model_lr.predict(user_data)[0]

        st.markdown("---")
        st.subheader("📋 Model Guidance Summary")

        if predicted_state == "Suicidal":
            st.error(
                "🔮 **Primary Profile Classification: Suicidal Risk Indicator**\n\nThe model identified linguistic patterns heavily associated with the Suicidal class partition in its training data.")
        elif predicted_state == "Depression":
            st.warning(
                "🔮 **Primary Profile Classification: Depressive Framework**\n\nThe model identified linguistic patterns heavily associated with the Depression class partition in its training data.")
        elif predicted_state == "Anxiety":
            st.info(
                "🔮 **Primary Profile Classification: Anxiety / Panic Framework**\n\nThe model identified linguistic patterns heavily associated with the Anxiety/Panic class partition in its training data.")
        else:
            st.success(
                "🔮 **Primary Profile Classification: Balanced Baseline Processing**\n\nThe inputs lack the specific structural semantic combinations that correlate with clinical risk groupings.")

        st.write("**Prediction Confidence Distribution Profile:**")
        st.text(
            f"⚡ Anxiety Markers:      {prob_dict.get('Anxiety', 0.0):.1f}%\n"
            f"📉 Depressive Symptoms:  {prob_dict.get('Depression', 0.0):.1f}%\n"
            f"🚨 Severe Crisis Risk:    {prob_dict.get('Suicidal', 0.0):.1f}%\n"
            f"🌱 Baseline Processing:  {prob_dict.get('Normal', 0.0):.1f}%"
        )

        st.markdown("---")
        st.subheader("🧠 Model Feature Weights Analysis")
        feature_names = vectorizer.get_feature_names_out()
        words_in_input = [word for word in user_text.lower().split() if word in feature_names]

        if words_in_input:
            word_impact = []
            predicted_class_idx = list(model_lr.classes_).index(predicted_state)
            for word in words_in_input:
                idx = list(feature_names).index(word)
                weight = model_lr.coef_[predicted_class_idx][idx]
                word_impact.append({"Feature Token": word, "Model Weight": round(weight, 4)})
            impact_df = pd.DataFrame(word_impact).sort_values(by="Model Weight", ascending=False)
            st.dataframe(impact_df, hide_index=True)
        else:
            st.write("Input strings do not contain features registered in the training vocabulary mapping.")

# --- TAB 2: BULK BATCH PROCESSING ---
with tab2:
    st.subheader("Multi-Entry Batch Upload Suite")
    st.sidebar.header("🎯 Dataset Screening Controls")
    search_keyword = st.sidebar.text_input("Search entries by specific word:", "")
    isolate_class = st.sidebar.selectbox(
        "Isolate by AI Diagnostic Classification:",
        ["All Records", "Normal", "Anxiety", "Depression", "Suicidal"]
    )

    uploaded_file = st.file_uploader("Choose a CSV file to process", type=["csv"], key="bulk_uploader")

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)
        if "text" in bulk_df.columns:
            bulk_df = bulk_df.dropna(subset=["text"])
            bulk_vectors = vectorizer.transform(bulk_df["text"])
            bulk_df["AI_Predicted_Risk"] = model_lr.predict(bulk_vectors)

            filtered_df = bulk_df.copy()
            if search_keyword:
                filtered_df = filtered_df[filtered_df["text"].str.contains(search_keyword, case=False, na=False)]
            if isolate_class != "All Records":
                filtered_df = filtered_df[filtered_df["AI_Predicted_Risk"] == isolate_class]

            st.markdown("---")
            st.success(
                f"Successfully parsed data pipeline. Showing {len(filtered_df)} matching records out of {len(bulk_df)} total entries.")

            if not filtered_df.empty:
                st.subheader("📊 Filtered Cohort Distribution")
                summary_counts = filtered_df["AI_Predicted_Risk"].value_counts()
                pie_fig = px.pie(names=summary_counts.index, values=summary_counts.values, color=summary_counts.index,
                                 color_discrete_map=color_map, hole=0.4)
                st.plotly_chart(pie_fig, width="stretch")

                st.subheader("📋 Flagged Records Data Suite")
                st.dataframe(filtered_df[["text", "AI_Predicted_Risk"]], hide_index=True)
        else:
            st.error("❌ Error: Could not find a column named 'text' in this file.")

# --- 🔬 TAB 3: SCIENTIFIC MODEL DIAGNOSTICS ---
with tab3:
    st.subheader("🔬 Comparative Model Performance Diagnostics")
    st.write(
        "Controlled engineering benchmarks comparing foundational Naive Bayes mapping against our optimized multi-class Logistic Regression pipeline using the hidden 20% test partition vault.")

    preds_lr = model_lr.predict(X_test)
    preds_nb = model_nb.predict(X_test)

    f1_lr = f1_score(y_test, preds_lr, average="macro") * 100
    f1_nb = f1_score(y_test, preds_nb, average="macro") * 100

    metrics_df = pd.DataFrame({
        "Algorithmic Architecture": ["CountVectorizer + Multinomial Naive Bayes (CS50 Baseline)",
                                     "TF-IDF + Bigrams + Logistic Regression (SentX Optimized)"],
        "Macro F1-Score (Balanced Grade)": [f"{f1_nb:.2f}%", f"{f1_lr:.2f}%"],
        "Pipeline Benchmark Status": ["Standard baseline tracker", "Highly optimized research build"]
    })
    st.dataframe(metrics_df, hide_index=True)

    st.markdown("---")
    st.subheader("📋 Granular Logistic Regression Validation Report")
    report_dict = classification_report(y_test, preds_lr, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().iloc[:-3]
    st.dataframe(report_df.style.format("{:.1%}", subset=["precision", "recall", "f1-score"]),
                 use_container_width=False)

    # 🛠️ NEW UPGRADE: Recommendation #3 (Error Analysis Sandbox)
    st.markdown("---")
    st.subheader("🕵️‍♂️ Pipeline Error Analysis Sandbox")
    st.write(
        "Below are real instances where the optimized model's prediction conflicted with the true data label. Investigating these anomalies highlights where psychological vocabularies naturally overlap.")

    # Create a dataframe of the testing vault to find mistakes
    error_analysis_df = pd.DataFrame({
        "Text Record Context": text_test,
        "True Human Label": y_test,
        "AI Fallible Prediction": preds_lr
    })

    # Isolate only rows where the True Label does NOT match the AI Prediction
    failures_df = error_analysis_df[
        error_analysis_df["True Human Label"] != error_analysis_df["AI Fallible Prediction"]]

    # Display the top 5 mistakes clearly
    st.dataframe(failures_df.head(5), hide_index=True)

# === Footnote Disclaimer ===
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption(
    "⚠️ **Academic Research & Ethical Disclaimer:** SentX is an experimental NLP classification system, not a medical diagnostic utility.")
