
import re
import os
import io
import docx2txt
import spacy
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,flash,session
from flask_sqlalchemy import SQLAlchemy
from flask import Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
import nltk
from flask import send_file, abort, send_from_directory
from flask import jsonify, request
import fitz 



nltk.download('punkt')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads2'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy()

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'



class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False)  

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_result.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    user = db.relationship('User', backref='ratings')
    analysis = db.relationship('AnalysisResult', backref='ratings')

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    course_title = db.Column(db.String(200), nullable=False)
    course_lecturer = db.Column(db.String(200), nullable=False)
    course_material = db.Column(db.String(300), nullable=False)
    course_outline = db.Column(db.String(300), nullable=False)
    total_words = db.Column(db.Integer, nullable=False)
    total_sentences = db.Column(db.Integer, nullable=False)
    fre_score = db.Column(db.Float, nullable=False)
    cc_score = db.Column(db.Float, nullable=False)
    topic_coverage_score = db.Column(db.Float, nullable=False)
    cctc_enhanced_fre = db.Column(db.Float, nullable=False)
    gunfog_score = db.Column(db.Float, nullable=False)
    tfidf_score = db.Column(db.Float, nullable=False)
    lsa_score = db.Column(db.Float, nullable=False)
    smog_score = db.Column(db.Float, nullable=False)
    course_material_filename = db.Column(db.String(255), nullable=False)
    course_outline_filename = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class CourseMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_result.id'), nullable=False)  # Link to AnalysisResult
    material_filename = db.Column(db.String(255), nullable=False)
    outline_filename = db.Column(db.String(255), nullable=True)
    analysis = db.relationship('AnalysisResult', backref='materials')


nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        file_stream = io.BytesIO(f.read())  
        with fitz.open(stream=file_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    return text



def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    return ' '.join(tokens)

def compute_flesch_reading_ease(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(len(re.findall(r'[aeiouy]+', word)) for word in words)
    if num_words == 0 or num_sentences == 0:
        return 0
    fre_score = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (num_syllables / num_words))
    fre_score = max(0, min(fre_score, 100))  # Cap FRE between 0 and 100
    return round(fre_score, 2)


def compute_conceptual_clarity(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    avg_score = np.mean(cosine_sim_matrix.data) if cosine_sim_matrix.nnz else 0
    avg_score = max(0, min(avg_score, 1))  # Cap CC between 0 and 1
    return round(avg_score, 2)


def compute_topic_coverage(material_text, syllabus_text):
    texts = [material_text, syllabus_text]
    tokenized_texts = [[word for word in word_tokenize(doc.lower()) if word.isalnum()] for doc in texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=25, random_state=42)
    topic_distribution = lda.get_document_topics(corpus[0], minimum_probability=0)
    syllabus_distribution = lda.get_document_topics(corpus[1], minimum_probability=0)
    similarity_score = sum(
        min(dict(topic_distribution).get(topic, 0), dict(syllabus_distribution).get(topic, 0))
        for topic, _ in topic_distribution
    ) * 100
    return round(min(similarity_score, 100), 2)  # Cap TC at 100


def compute_cctc_enhanced_fre(fre_score, cc_score, topic_coverage_score):
    cc_score = min(cc_score * 10, 100)  # CC is scaled to 0–100
    eFRE = (fre_score * 0.4) + (cc_score * 0.3) + (topic_coverage_score * 0.3)
    return round(min(eFRE, 100), 2)  # Cap enhanced FRE at 100


def compute_gunning_fog(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    complex_words = len([word for word in words if len(word) > 2 and len(re.findall(r'[aeiouy]+', word)) >= 3])
    if len(sentences) == 0:
        return 0
    score = 0.4 * ((len(words) / len(sentences)) + (100 * (complex_words / len(words))))
    return round(min(score, 20), 2)  # Cap at 20 for readability


def compute_tfidf_score(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    non_zero_values = tfidf_matrix.data
    score = np.mean(non_zero_values) if non_zero_values.size else 0
    return round(min(score, 1.5), 2)  # Cap TF-IDF average score at 1.5


def compute_lsa_score(text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([text])
    if tfidf_matrix.shape[1] < 2:
        return 0
    svd = TruncatedSVD(n_components=min(10, tfidf_matrix.shape[1] - 1))
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    score = np.mean(lsa_matrix)
    return round(min(score, 1.5), 2)  # Cap LSA score at 1.5


def compute_smog_score(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    polysyllabic_words = len([word for word in words if len(re.findall(r'[aeiouy]+', word)) >= 3])
    if len(sentences) == 0:
        return 0
    score = 1.043 * (30 * (polysyllabic_words / len(sentences))) ** 0.5 + 3.1291
    return round(min(score, 20), 2)  # Cap SMOG score at 20

def get_material_by_id(material_id):
    # Query the AnalysisResult table by material ID
    material = AnalysisResult.query.get(material_id)
    
    return material


@login_manager.user_loader
def load_user(user_id):
    print("Loading user:", user_id)  # Debugging
    return User.query.get(int(user_id))



@app.route('/')
def home():

    return render_template('index.html')  # Always return a valid response



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            session['username'] = user.username  
            session['role'] = user.role  
            flash('Login successful!', 'success')
            print(f"User {username} logged in successfully!")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Try again.', 'danger')
            print("Login failed!")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])  # Hash the password
        role = request.form['role']
        try:
            new_user = User(username=username, password=password, role=role)
            db.session.add(new_user)
            db.session.commit()  
        except Exception as e:
            db.session.rollback()  
            flash("Registration failed. Try again.")
            return redirect(url_for('register'))
        finally:
            db.session.close()  

        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/rate_document', methods=['GET'])
@login_required
def rate_document():
    materials = AnalysisResult.query.all() 

    if not materials:
        flash("No course materials available for rating.", "warning")
        return redirect(url_for('dashboard'))  
    return render_template('rate_document.html', materials=materials)

@app.route('/submit_rating/<int:analysis_id>', methods=['POST'])
def submit_rating(analysis_id):
    rating = request.form.get('rating')
    if rating:
        new_rating = Rating(analysis_id=analysis_id, user_id=current_user.id, rating=int(rating))
        db.session.add(new_rating)
        db.session.commit()     
        return jsonify({"message": "Rating submitted successfully!"}), 200
    return jsonify({"error": "Please select a rating"}), 400

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads2')

@app.route('/view_document/<int:material_id>/<doc_type>', methods=['GET'])
def view_document(material_id, doc_type):
    material = get_material_by_id(material_id)
    
    if not material:
        return "Material not found", 404
    
    if doc_type == 'material':
        course_text = material.course_material  # Access course_material directly
        filename = material.course_material_filename  # Access the material filename
    elif doc_type == 'outline':
        course_text = material.course_outline  # Access course_outline directly
        filename = material.course_outline_filename  # Access the outline filename
    else:
        return "Invalid document type", 400
    return render_template('view_document.html', course_text=course_text, filename=filename, material_id=material.id) 
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = current_user.username
    results = AnalysisResult.query.filter_by(username=username).all()  
    for res in results:
        print(f"Analysis ID: {res.id}, Course Title: {res.course_title}")

    if request.method == 'POST':
        file = request.files.get('file')
        syllabus_file = request.files.get('syllabus_file')
        course_lecturer = request.form.get('course_lecturer')
        allowed_extensions = {'docx', 'pdf'}
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        syllabus_ext = syllabus_file.filename.rsplit('.', 1)[1].lower()

        if not file or not syllabus_file or file_ext not in allowed_extensions or syllabus_ext not in allowed_extensions:
            flash("Error: Please upload valid .docx or .pdf files!", "danger")
            return redirect(url_for('dashboard'))
        else:
            flash('File uploaded successfully!', 'success')

        

        
        existing_result = AnalysisResult.query.filter_by(
            username=username, 
            course_title=file.filename,
            course_lecturer=course_lecturer
        ).first()

        if existing_result:
            flash("This document has already been analyzed!", "warning")
        else:
            # Save the files
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            syllabus_path = os.path.join(app.config['UPLOAD_FOLDER'], syllabus_file.filename)
            file.save(file_path)
            syllabus_file.save(syllabus_path)

            # Extract text based on file type
            if file_ext == 'pdf':
                course_material_text = extract_text_from_pdf(file_path)
            else:
                course_material_text = docx2txt.process(file_path)

            if syllabus_ext == 'pdf':
                course_outline_text = extract_text_from_pdf(syllabus_path)
            else:
                course_outline_text = docx2txt.process(syllabus_path)

            # Compute scores
            fre_score = compute_flesch_reading_ease(course_material_text)
            cc_score = compute_conceptual_clarity(course_material_text)
            topic_coverage_score = compute_topic_coverage(course_material_text, course_outline_text)
            cctc_enhanced_fre = compute_cctc_enhanced_fre(fre_score, cc_score, topic_coverage_score)
            gunfog_score = compute_gunning_fog(course_material_text)
            tfidf_score = compute_tfidf_score(course_material_text)
            lsa_score = compute_lsa_score(course_material_text)
            smog_score = compute_smog_score(course_material_text)

            # Create new result
            new_result = AnalysisResult(
                username=username,
                user_id=current_user.id,
                course_title=file.filename,
                course_lecturer=course_lecturer,
                course_material=course_material_text,
                course_outline=course_outline_text,
                total_words=len(course_material_text.split()),
                total_sentences=len(sent_tokenize(course_material_text)),
                fre_score=fre_score,
                cc_score=cc_score,
                topic_coverage_score=topic_coverage_score,
                cctc_enhanced_fre=cctc_enhanced_fre,
                gunfog_score=gunfog_score,
                tfidf_score=tfidf_score,
                lsa_score=lsa_score,
                smog_score=smog_score,
                course_material_filename=file.filename,  # Store the material filename
                course_outline_filename=syllabus_file.filename  # Store the outline filename
            )

            # Commit the new result to the database
            db.session.add(new_result)
            db.session.commit()

            flash("File uploaded and analyzed successfully!", "success")

    results = AnalysisResult.query.filter_by(username=username).all()
    return render_template('dashboard.html', results=results)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    users = User.query.all()
    documents = AnalysisResult.query.all()  
    return render_template('admin_dashboard.html', users=users, documents=documents)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
   