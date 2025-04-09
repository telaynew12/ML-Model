import streamlit as st
from team_logic import (
    load_data, infer_skills, extract_skills, compute_suitability, select_diverse_team
)

# Load and prepare data
@st.cache_data
def cached_data():
    df = load_data()
    df, _, _ = infer_skills(df)
    return df

df = cached_data()

# Predefined project ideas
predefined_projects = {
    "Secure Software Platform": "Develop a secure web application with robust cybersecurity measures.",
    "Enterprise Web Portal": "Design and develop a scalable web software solution for enterprise clients.",
    "Network Security Upgrade": "Implement robust cybersecurity protocols to protect network infrastructure.",
    "Smart City Dashboard": "Develop a web-based dashboard with software integration and cybersecurity for smart city infrastructure.",
    "TeamSync Platform": "Build a collaborative software tool to enhance team management and leadership."
}

st.title("AI-Powered Team Formation")

tab1, tab2 = st.tabs(["Select Existing Project", "Add New Project"])

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

with tab2:
    new_project_name = st.text_input("New Project Name", key="new_project")
    new_project_description = st.text_area("New Project Description", key="desc_new")

if selected_project:
    project_name = selected_project
    description_to_use = project_description
else:
    project_name = new_project_name
    description_to_use = new_project_description

team_size = st.slider("Team Size", min_value=1, max_value=10, value=5)

if st.button("Form Team"):
    if not project_name or not description_to_use:
        st.error("Please provide a project name and description.")
    else:
        required_skills = extract_skills(description_to_use, df)
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

        if not alternates.empty:
            st.subheader("Alternate Candidates")
            st.dataframe(alternates[['user_id', 'full_name', 'team_name', 'suitability_score']])

if st.checkbox("Show Skill Clusters"):
    st.write("Employee Skill Clusters:")
    st.dataframe(df.groupby('skill_cluster').agg({'full_name': 'count', 'average_kpi_score_q5': 'mean'}))
