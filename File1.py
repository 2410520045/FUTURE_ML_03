import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Sample Data Setup
# -------------------------------------------------------------------
job_descriptions = {
    "Data Scientist": """
    Looking for a Data Scientist with strong skills in Python, Machine Learning, 
    SQL, Scikit-learn, Data Analysis, TF-IDF, NLP, and Deep Learning.
    """,
    "Software Engineer": """
    Required Software Engineer proficient in Java, C++, Python, Data Structures, 
    Algorithms, Git, Object-Oriented Programming, and SQL.
    """
}

resumes_data = [
    {"candidate_id": "C101", "name": "Alice Johnson", "resume": "Experienced Data Scientist skilled in Python, Machine Learning, SQL, Scikit-learn, TF-IDF, and NLP applications."},
    {"candidate_id": "C102", "name": "Bob Smith", "resume": "Software Engineer with expertise in Java, C++, Data Structures, Algorithms, Git, and Object-Oriented Programming."},
    {"candidate_id": "C103", "name": "Charlie Brown", "resume": "Data Analyst with knowledge of Python, SQL, Excel, and introductory Machine Learning techniques."},
    {"candidate_id": "C104", "name": "Diana Prince", "resume": "Full Stack Engineer proficient in Python, JavaScript, HTML, CSS, React, SQL, and Git."}
]

skills_db = ["python", "machine learning", "sql", "scikit-learn", "tf-idf", "nlp", "deep learning", 
             "java", "c++", "data structures", "algorithms", "git", "excel", "react", "html", "css"]

# -------------------------------------------------------------------
# 2. Text Cleaning & Skill Extraction
# -------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s#\+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skills_list):
    text_cleaned = clean_text(text)
    extracted = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text_cleaned)]
    return list(set(extracted))

# -------------------------------------------------------------------
# 3. Resume Screening & Ranking
# -------------------------------------------------------------------
def screen_resumes(target_role):
    jd_text = job_descriptions[target_role]
    jd_clean = clean_text(jd_text)
    jd_skills = set(extract_skills(jd_clean, skills_db))
    
    clean_resumes = [clean_text(r["resume"]) for r in resumes_data]
    
    # TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_clean] + clean_resumes)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    results = []
    for idx, cand in enumerate(resumes_data):
        cand_skills = set(extract_skills(clean_resumes[idx], skills_db))
        matched_skills = list(cand_skills.intersection(jd_skills))
        missing_skills = list(jd_skills - cand_skills)
        match_score = round(float(scores[idx]) * 100, 2)
        
        results.append({
            "Candidate ID": cand["candidate_id"],
            "Name": cand["name"],
            "Match Score (%)": match_score,
            "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
            "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
        })
        
    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
    return df

# Run Screening for Data Scientist role
ranked_df = screen_resumes("Data Scientist")
print("=== Candidate Ranking for Data Scientist ===")
print(ranked_df.to_string(index=False))

ranked_df.to_csv("candidate_rankings.csv", index=False)

# -------------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.barh(ranked_df["Name"], ranked_df["Match Score (%)"], color='skyblue')
plt.xlabel("Match Score (%)")
plt.title("Candidate Scores - Data Scientist Role")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("01_candidate_scores.png")
plt.close()