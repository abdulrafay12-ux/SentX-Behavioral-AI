# SentX: Experimental NLP Text Classification Dashboard

SentX is an open-source, web-based software prototype designed to analyze textual inputs and evaluate semantic data patterns matching four categories: Normal, Anxiety, Depression, and Suicidal text logs. By training a multi-class linear classification pipeline on an external balanced corpus of text posts, the system maps unstructured linguistic phrases onto statistical probability distributions to help explore automated text categorization.

**Live Application URL:** [https://sentx-behavioral-ai-gvcqqgpnzz9ppvvdvwcur4.streamlit.app/

---

## 🎯 Strategic Features
*   **Single String Screening:** An individual input text block that outputs color-coded probability distribution analytics.
*   **Bulk CSV Batch Processing:** A tabular file processing pipeline allowing users to upload a `.csv` spreadsheet log and screen multiple entries simultaneously.
*   **Linguistic Model Feature Weights:** A transparency utility that pulls global model coefficient statistics to show which tokens strongly correlate with the active prediction class.
*   **Pipeline Error Analysis Sandbox:** An integrated diagnostic portal displaying live instances where the machine learning model's prediction conflicted with the true data label, highlighting semantic overlaps in the vocabulary.

---

## 🔬 Model Development History
The system was built through an iterative development process to evaluate how changes in feature extraction and model architectures impact classification performance:

1.  **Baseline Prototype (Stage 1):** Built a standard binary classifier to distinguish between simple text labels using an 8-row control data frame. This baseline utilized `CountVectorizer` for raw word token frequencies and a `Multinomial Naive Bayes` algorithm. This layout evaluated tokens in isolation, meaning it was unable to capture localized phrase parameters or simple word negation context (e.g., treating *"not fine"* and *"fine"* as highly similar vectors).
2.  **Optimized Architecture (Stage 2):** Upgraded the configuration to a multi-class **Logistic Regression** classifier. To improve context window tracking, I implemented a **TF-IDF Vectorizer** configured with `ngram_range=(1, 2)`. This setting directs the pipeline to process adjacent word pairs (**bi-grams**) alongside single words (unigrams), allowing the model to register localized word-order patterns like *"feel trapped"* or *"not coping"*.

---

## 📊 Evaluation & Empirical Performance
To test the models rigorously, the source dataset was partitioned using a standard 80/20 train-test split via `scikit-learn` (`train_test_split`). This isolates exactly 199 random samples inside a hidden testing partition vault to evaluate performance on data unseen during training.

### Controlled Algorithmic Benchmarks
*   **Baseline (CountVectorizer + Naive Bayes):** `47.19%` Macro F1-Score
*   **Optimized Pipeline (TF-IDF + Bigrams + Logistic Regression):** **`63.28%` Macro F1-Score**

### Granular Logistic Regression Validation Matrix
*Evaluation metrics generated across the 199 unseen test sample distributions:*

| Classification Category | Precision | Recall | F1-Score | Support Samples |
| :--- | :--- | :--- | :--- | :--- |
| **Anxiety** | 53.1% | 59.1% | 55.9% | 44 |
| **Depression** | 55.1% | 54.0% | 54.5% | 50 |
| **Normal** | 79.3% | 88.2% | 83.5% | 51 |
| **Suicidal** | 64.6% | 54.5% | 59.2% | 55 |
| **Macro Average** | **63.0%** | **63.9%** | **63.28%** | **199** |

---

## 📂 Dataset Provenance & Local Execution
The model was trained using an open-source, balanced text dataset sourced from Kaggle (`mental_health_combined_test.csv`), containing 992 downsampled entries partitioned into equal class segments for objective statistical evaluation.

### Dependencies
Dependencies are listed inside `requirements.txt`:
*   `streamlit`
*   `pandas`
*   `scikit-learn`
*   `plotly`

### Local Setup Instructions
To run this application dashboard locally on your desktop:
1.  Clone or download this repository folder.
2.  Open your command terminal inside the project root directory and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the interactive dashboard server:
    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Limitations & Development Process Statement
*   **Experimental Scope:** SentX is a conceptual NLP text classification system designed strictly for independent academic research. It is **not** a clinically validated diagnostic tool, medical triage framework, or crisis evaluation engine. Predictions entirely reflect mathematical probability weights learned from its specific source training data and can contain false positives, categorization errors, or dataset-specific alignment biases.
*   **Development Process:** The conceptual framework, personal motivation, and software requirements were designed independently. A generative AI assistant was utilized as a collaborative partner to assist with Python syntax writing, debugging array matrix constraints, and structuring technical markdown documentation templates. All presented methods and mathematical evaluation matrices have been verified for accuracy.
