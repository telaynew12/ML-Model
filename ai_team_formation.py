import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_data.csv")
    return df

df = load_data()

# Load NLP model
nlp = spacy.load("en_core_web_sm")
skill_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess data for clustering
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

# Infer skills with clustering
@st.cache_resource
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

df, kmeans, scaler = infer_skills(df)

# Enhanced skill extraction with weights
def extract_skills(description):
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

# Improved suitability scoring
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

# Ensure team diversity with flexible matching and alternates
def select_diverse_team(df, team_size, required_skills):
    df_sorted = df.sort_values('suitability_score', ascending=False)
    selected = []
    team_counts = {}
    available_teams = df['team_name'].unique().tolist()
    st.write(f"Available Teams in Data: {available_teams}")

    # Priority teams with flexible matching
    priority_teams = set()
    for skill in required_skills:
        if 'development' in skill.lower() or 'develop' in skill.lower() or 'software' in skill.lower() or 'web' in skill.lower():
            priority_teams.update([t for t in available_teams if 'Software Development' in t])
        if 'cybersecurity' in skill.lower():
            priority_teams.update([t for t in available_teams if 'Cyber Security' in t])
    st.write(f"Priority Teams: {list(priority_teams)}")

    # First pass: Ensure at least one from each priority team if available
    for team in priority_teams:
        team_df = df_sorted[df_sorted['team_name'] == team]
        if not team_df.empty:
            candidate = team_df.iloc[0]
            selected.append(candidate)
            team_counts[team] = team_counts.get(team, 0) + 1
        else:
            st.warning(f"No employees found for team: {team}")

    # Second pass: Fill remaining slots with top scorers, limiting team dominance
    remaining = team_size - len(selected)
    for _, row in df_sorted.iterrows():
        if len(selected) >= team_size:
            break
        team = row['team_name']
        if team not in team_counts or team_counts[team] < max(2, team_size // max(1, len(priority_teams))):
            if row['user_id'] not in [x['user_id'] for x in selected]:
                selected.append(row)
                team_counts[team] = team_counts.get(team, 0) + 1

    # Select alternates from remaining candidates
    remaining_df = df_sorted[~df_sorted['user_id'].isin([x['user_id'] for x in selected])]
    alternates = remaining_df.head(3)  # Top 3 alternates

    return pd.DataFrame(selected), alternates

# Sample predefined projects
predefined_projects = {
    "Secure Software Platform": "Develop a secure web application with robust cybersecurity measures.",
    "Enterprise Web Portal": "Design and develop a scalable web software solution for enterprise clients.",
    "Network Security Upgrade": "Implement robust cybersecurity protocols to protect network infrastructure.",
    "Smart City Dashboard": "Develop a web-based dashboard with software integration and cybersecurity for smart city infrastructure.",
    "TeamSync Platform": "Build a collaborative software tool to enhance team management and leadership."
}

# Streamlit app
st.title("AI-Powered Team Formation")

# Tabs for selecting or adding a project
tab1, tab2 = st.tabs(["Select Existing Project", "Add New Project"])

# Tab 1: Dropdown for existing projects
with tab1:
    selected_project = st.selectbox(
        "Select a Project",
        options=[""] + list(predefined_projects.keys()),
        index=0,
        help="Search and select an existing project."
    )
    if selected_project:
        project_description = st.text_area(
            "Project Description (Editable)",
            value=predefined_projects[selected_project],
            key="desc_existing"
        )
    else:
        project_description = st.text_area(
            "Project Description (Editable)",
            value="",
            key="desc_existing_empty"
        )

# Tab 2: Text input for new project
with tab2:
    new_project_name = st.text_input("New Project Name", key="new_project")
    new_project_description = st.text_area("New Project Description", key="desc_new")

# Determine which project name and description to use
if selected_project:
    project_name = selected_project
    description_to_use = project_description
else:
    project_name = new_project_name
    description_to_use = new_project_description

# Team size input
team_size = st.slider("Team Size", min_value=1, max_value=10, value=5)

# Form team button
if st.button("Form Team"):
    if not project_name or not description_to_use:
        st.error("Please provide a project name and description.")
    else:
        required_skills = extract_skills(description_to_use)
        st.write(f"Extracted Skills: {list(required_skills.keys())}")
        df_with_scores = compute_suitability(df.copy(), required_skills)
        recommended_team, alternates = select_diverse_team(df_with_scores, team_size, required_skills)
        recommended_team = recommended_team[
            ['user_id', 'full_name', 'team_name', 'inferred_skills', 'suitability_score',
             'average_kpi_score_q5', 'average_engagement_score_q5']
        ]
        st.subheader(f"Recommended Team for {project_name}")
        st.dataframe(recommended_team)
        st.bar_chart(recommended_team.set_index('full_name')['suitability_score'])

        # Display alternate candidates
        if not alternates.empty:
            st.subheader("Alternate Candidates")
            st.dataframe(alternates[['user_id', 'full_name', 'team_name', 'suitability_score']])

# Optional skill clusters
if st.checkbox("Show Skill Clusters"):
    st.write("Employee Skill Clusters:")
    st.dataframe(df.groupby('skill_cluster').agg({'full_name': 'count', 'average_kpi_score_q5': 'mean'}))