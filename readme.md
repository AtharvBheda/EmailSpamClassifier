# **Email Spam Classifier**

This project is an **Email Spam Classifier** built using Python, NLP, and Machine Learning. It predicts whether an email is spam or not based on its content, utilizing a trained classification model and an interactive web interface.

---

## **Features**
- **Input Options**:
  - Manually type the email content.
  - Upload a PDF or text file for email classification.
- **Text Preprocessing**:
  - Converts text to lowercase.
  - Removes punctuation and stopwords.
  - Applies stemming using the PorterStemmer.
- **Spam Prediction**:
  - Displays whether an email is classified as "Spam" or "Not Spam."
  - Provides processed text for user understanding.
- **Interactive Interface**:
  - Built with **Streamlit** for easy use.
  - Displays word count and additional statistics.

---

## **Dataset**
The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains labeled text messages (spam or not spam). It includes:
- **5574 samples**: 
  - 4827 ham (not spam)
  - 747 spam

The dataset has been preprocessed to train the machine learning model used in this project.

---

## **How It Works**
1. **Preprocessing**:
   - Tokenizes and cleans the text.
   - Removes non-alphanumeric characters and stopwords.
   - Applies stemming to reduce words to their root forms.
2. **Text Vectorization**:
   - Converts processed text into numerical features using a **TF-IDF vectorizer**.
3. **Prediction**:
   - Uses a trained **Naive Bayes Classifier** to predict if an email is "Spam" or "Not Spam."
4. **Result Display**:
   - Outputs the classification result with a color-coded background:
     - **Red**: Spam
     - **Green**: Not Spam

---

## **Technologies Used**
- **Python**: Programming language.
- **Streamlit**: Web framework for the interactive interface.
- **NLTK**: For natural language processing and text preprocessing.
- **PyPDF2**: For extracting text from PDF files.
- **Scikit-learn**: For vectorization (TF-IDF) and model building.

---

## **Project Structure**
```
SpamClassifier/
├── app.py                 # Streamlit app script
├── sms-spam-detection.ipynb # Jupyter Notebook for training and EDA
├── vectorizer.pkl         # TF-IDF vectorizer (pickle file)
├── model.pkl              # Trained Naive Bayes model (pickle file)
├── spam.csv               # Dataset used for training/testing
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## **Installation and Usage**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/SpamClassifier.git
cd SpamClassifier
```

### **Step 2: Create a Virtual Environment**
```bash
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Streamlit App**
```bash
streamlit run app.py
```

### **Step 5: Access the App**
- Open the URL displayed in the terminal (usually `http://localhost:8501`).
- Use the interface to classify emails.

---

## **Usage**
1. **Enter Email Content**:  
   Type the email manually or upload a text/PDF file.
2. **Submit for Classification**:  
   Click the "Submit" button to get the classification result.

---

## **Demo**
![image](https://github.com/user-attachments/assets/02eb248e-f0ff-4172-bc14-5aceeaed8298)
### **Ham Mail:**
![image](https://github.com/user-attachments/assets/fa8ee6e9-cfac-4ead-8a26-5d47e001ca06)
### **Spam Mail:**


---

## **Model Training Details**
1. **Model**: Multinomial Naive Bayes.
2. **Vectorization**: TF-IDF vectorizer for feature extraction.
3. **Performance**:
   - Accuracy: 97.5% on the test set.
   - Precision and Recall: High for spam detection.

---

## **Contribution**
Feel free to fork the repository and submit pull requests. Contributions are welcome to improve the app or add new features!

---
