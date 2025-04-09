import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load NLP models
nlp = spacy.load("en_core_web_sm")
skill_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
def load_data():
    df = pd.read_csv("data/employee_data.csv")
    return df

# Preprocess data
def preprocess_data(df):
    features = [
        'average_weekly_score_q5', 'average_manager_score_q5', 'average_engagement_score_q5',
        'average_kpi_score_q5', 'average_okr_score_q5', 'AppCount_q5', 'ReprimandCount_q5',
        'sum_tardy_q5', 'sum_absent_q5', 'year_of_service_q5'
    ]
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler

# Infer skill clusters
def infer_skills(df):
    X_scaled, features, scaler = preprocess_data(df)
    kmeans = KMeans(n_clusters=7, random_state=42)
    df['skill_cluster'] = kmeans.fit_predict(X_scaled)
    cluster_labels = {
        0: "High-Performing Technical",
        1: "Collaborative Leader",
        2: "Consistent Technical Performer",
        3: "Low Engagement",
        4: "Newcomer Potential",
        5: "High Engagement Specialist",
        6: "Balanced Performer"
    }
    df['inferred_skills'] = df['skill_cluster'].map(cluster_labels)
    return df, kmeans, scaler

# Extract skills from project description
def extract_skills(description, df):
    doc = nlp(description)
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"] and token.text.lower() not in ["application"]]
    predefined_skills = {
        'software': 1.0, 'development': 1.5, 'develop': 1.5, 'cybersecurity': 1.5, 'management': 1.0,
        'collaboration': 1.0, 'leadership': 1.0, 'web': 1.2
    }
    extracted_skills = {}
    for kw in keywords:
        for skill, weight in predefined_skills.items():
            if skill in kw:
                extracted_skills[skill] = weight
    if not extracted_skills:
        for team in df['team_name'].unique():
            if team.lower() in description.lower():
                extracted_skills[team.lower()] = 1.0
    return extracted_skills if extracted_skills else {"General": 1.0}

# Compute suitability scores
def compute_suitability(df, required_skills):
    skill_names = list(required_skills.keys())
    skill_weights = list(required_skills.values())
    req_skill_emb = skill_model.encode(skill_names)
    req_skill_emb_weighted = np.average(req_skill_emb, weights=skill_weights, axis=0).reshape(1, -1)
    employee_profiles = df['inferred_skills'] + " " + df['team_name']
    emp_emb = skill_model.encode(employee_profiles.tolist())
    similarities = cosine_similarity(emp_emb, req_skill_emb_weighted)
    df['suitability_score'] = similarities.flatten() * 70
    df['metric_score'] = (
        0.4 * df['average_kpi_score_q5'] +
        0.3 * df['average_engagement_score_q5'] +
        0.2 * df['average_okr_score_q5'] +
        0.1 * df['average_manager_score_q5']
    ).clip(upper=100)
    df['suitability_score'] += df['metric_score'] * 0.3
    df['suitability_score'] -= (df['ReprimandCount_q5'] * 3 + df['sum_tardy_q5'] * 2 + df['sum_absent_q5'] * 5)
    df['suitability_score'] = df['suitability_score'].clip(lower=0)
    return df

# Team selection
def select_diverse_team(df, team_size, required_skills):
    df_sorted = df.sort_values('suitability_score', ascending=False)
    selected = []
    team_counts = {}
    available_teams = df['team_name'].unique().tolist()

    priority_teams = set()
    for skill in required_skills:
        if 'development' in skill.lower() or 'develop' in skill.lower() or 'software' in skill.lower() or 'web' in skill.lower():
            priority_teams.update([t for t in available_teams if 'Software Development' in t])
        if 'cybersecurity' in skill.lower():
            priority_teams.update([t for t in available_teams if 'Cyber Security' in t])

    for team in priority_teams:
        team_df = df_sorted[df_sorted['team_name'] == team]
        if not team_df.empty:
            candidate = team_df.iloc[0]
            selected.append(candidate)
            team_counts[team] = team_counts.get(team, 0) + 1

    for _, row in df_sorted.iterrows():
        if len(selected) >= team_size:
            break
        team = row['team_name']
        if team not in team_counts or team_counts[team] < max(2, team_size // max(1, len(priority_teams))):
            if row['user_id'] not in [x['user_id'] for x in selected]:
                selected.append(row)
                team_counts[team] = team_counts.get(team, 0) + 1

    remaining_df = df_sorted[~df_sorted['user_id'].isin([x['user_id'] for x in selected])]
    alternates = remaining_df.head(3)
    return pd.DataFrame(selected), alternates

# Load and update existing projects
def load_projects():
    return pd.read_csv("existing_projects.csv")

def save_projects(projects_df):
    projects_df.to_csv("existing_projects.csv", index=False)

# Add a new project with the selected team
def add_new_project(project_name, project_description, team_size, employee_data):
    projects_df = load_projects()
    new_project = {
        "project_name": project_name,
        "description": project_description,
        "team_size": team_size,
        "team_members": []  # Will be populated after forming the team
    }
    projects_df = projects_df.append(new_project, ignore_index=True)
    
    # Infer skills and form the team
    employee_data, _, _ = infer_skills(employee_data)
    required_skills = extract_skills(project_description, employee_data)
    employee_data = compute_suitability(employee_data, required_skills)
    team, alternates = select_diverse_team(employee_data, team_size, required_skills)
    
    # Save the team members for the project
    projects_df.at[projects_df.index[-1], "team_members"] = team["user_id"].tolist()
    
    # Save the updated projects
    save_projects(projects_df)
    
    return team, alternates

# Streamlit UI to create new project and team
st.title("Add New Project and Form Team")

# Input fields for the project
project_name = st.text_input("Project Name")
project_description = st.text_area("Project Description")
team_size = st.number_input("Team Size", min_value=1, max_value=10)

if st.button("Add New Project"):
    # Load employee data
    employee_data = load_data()
    
    # Add new project and form team
    team, alternates = add_new_project(project_name, project_description, team_size, employee_data)
    
    st.write(f"New project '{project_name}' added successfully.")
    st.write(f"Selected Team: {team[['user_id', 'team_name']]}")
    st.write(f"Alternates: {alternates[['user_id', 'team_name']]}")
