import streamlit as st
import string
import pickle
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and description
st.title("Email Spam Classifier")
st.write("Enter the email content below, and the system will predict whether it's Spam or Not Spam.")

# Sidebar for additional information
st.sidebar.title("Email Classification Features")
st.sidebar.write("""
This tool allows you to:
- Enter email content manually.
- Upload a PDF or text file with email content.
- Get word count and text statistics.
- Predict if the email is Spam or Not Spam.
""")

# File uploader for email text or PDF files
uploaded_file = st.sidebar.file_uploader("Upload a PDF or text file", type=['pdf', 'txt'])

# Input area for email text
input_email = ""

if uploaded_file is not None:
    # Check if the file is a PDF or TXT file
    if uploaded_file.name.endswith(".pdf"):
        input_email = extract_text_from_pdf(uploaded_file)
        st.text_area("Email Body (from PDF file)", input_email, height=250)
    else:
        input_email = uploaded_file.read().decode('utf-8')
        st.text_area("Email Body (from text file)", input_email, height=250)
else:
    input_email = st.text_area("Enter the email body manually", height=250, placeholder="Type your email content here...")

# Show word count statistics
if input_email:
    word_count = len(input_email.split())
    st.write(f"Word Count: {word_count} words")

# Button to trigger classification
if st.button('Submit'):
    with st.spinner('Processing...'):
        # Transform and predict
        transformed_email = transform_text(input_email)
        vector_input = tfidf.transform([transformed_email])
        result = model.predict(vector_input)[0]

        # Display the result with color-coded background
        if result == 1:
            st.markdown(
                "<div style='background-color: red; padding: 10px; border-radius: 5px;'>"
                "<h2 style='color: white;'>Result: This email is classified as Spam.</h2></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color: green; padding: 10px; border-radius: 5px;'>"
                "<h2 style='color: white;'>Result: This email is classified as Not Spam.</h2></div>",
                unsafe_allow_html=True
            )

        # Display transformed text for user understanding
        st.subheader("Processed Text for Model:")
        st.write(transformed_email)

# Footer with additional info
st.sidebar.info("This tool uses Natural Language Processing (NLP) and machine learning to classify emails.")
