# Skill_Match_AI:AI-Powered Resume Analysis & Career Optimization

Overview:

SkillMatch AI is a data science and NLP-powered project designed to automatically analyze and compare candidate resumes against job descriptions.
It calculates a match score, identifies matched and missing skills, and provides improvement suggestions for candidates.
Additionally, it integrates a Power BI dashboard for visualizing job-role match analytics, helping recruiters and candidates gain data-driven insights.

This project was developed as part of a data science , combining data preprocessing, NLP, and visualization into a real-time application.

Key Features:

1) Resume upload and parsing (PDF, DOCX, or TXT)

2) Automatic extraction of text and keywords

3) NLP-based job description matching using TF-IDF and cosine similarity

4) Skill extraction, matching, and missing skill identification

5) Interactive UI using Streamlit

6) Match score visualization using bar charts

7) Option to export results as CSV or PDF

Project Workflow:

1) Data Extraction:
Parse resume and job description files using PyPDF2 and python-docx.

2) Text Preprocessing:
Clean and normalize the text using NLTK/spaCy (tokenization, stopword removal, lemmatization).

3) Feature Engineering:
Apply TF-IDF vectorization to represent both documents numerically.

4) Similarity Computation:
Use cosine similarity to calculate the match score between the resume and job description.

5) Skill Analysis:
Identify matched and missing skills using NLP-based keyword extraction.

6) Visualization:
Display match results and skill comparison through Streamlit charts and Power BI dashboards.

7) Output Generation:
Allow users to download match results and insights as CSV or PDF files.

Tools and Technologies:

1) Python 3.10 – Core programming language used for data processing, NLP, and backend logic

2) Pandas & NumPy – For data cleaning, preprocessing, and numerical computation

3) Scikit-learn (TF-IDF, Cosine Similarity) – To measure text similarity between resumes and job descriptions

4) NLTK & spaCy – For Natural Language Processing tasks such as tokenization, stopword removal, and lemmatization

5) PyPDF2 & python-docx – To extract and parse text content from PDF and Word documents

6) Streamlit – For building an interactive and real-time web interface

7) Plotly – For visualizing match scores and skill comparisons through dynamic charts

8) CSV & PDF Export Modules – For generating downloadable reports of the match results
