import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords


resumes = pd.read_csv("resumes.csv")


with open("job_description.txt", "r") as file:
    job_description = file.read()


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)


clean_job = clean_text(job_description)


resumes["Cleaned_Resume"] = resumes["Resume"].apply(clean_text)


documents = [clean_job] + resumes["Cleaned_Resume"].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

resumes["Match_Score"] = similarity_scores

ranked_resumes = resumes.sort_values(by="Match_Score", ascending=False)


print("\nCandidate Ranking:\n")

for index, row in ranked_resumes.iterrows():
    print(f"{row['Name']} — Match Score: {round(row['Match_Score'], 2)}")