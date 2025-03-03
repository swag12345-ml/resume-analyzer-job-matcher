from flask import Flask, render_template, request, send_file
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Global variable to store results
results = []


# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


# Extract entities using spaCy NER
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names


@app.route('/', methods=['GET', 'POST'])
def index():
    global results  # Use global variable to store results
    results = []  # Reset results for each new request

    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        processed_resumes = []
        for resume_file in resume_files:
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)

            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text))

        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

        ranked_resumes = []
        for (names, emails, resume_text) in processed_resumes:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
            ranked_resumes.append((names, emails, similarity))

        ranked_resumes.sort(key=lambda x: x[2], reverse=True)
        results = ranked_resumes  # Store results globally for download

    return render_template('index.html', results=results)


@app.route('/download_csv')
def download_csv():
    global results  # Ensure access to the global results variable

    if not results:
        return "No data available for download", 400

    # Generate the CSV content
    csv_content = "Rank,Name,Email,Similarity\n"
    for rank, (names, emails, similarity) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        csv_content += "{},{},{},{}\n".format(rank, name, email, similarity)

    # Create a temporary file to store the CSV content
    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_content)

    # Send the file for download
    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")


if __name__ == '__main__':
    app.run(debug=True)
