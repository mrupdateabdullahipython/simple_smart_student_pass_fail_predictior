import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Page Configuration
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="wide"
)

# 2. Premium Cyber Dark CSS Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Glassmorphic Feature Input Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }
    
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: bold;
    }

    /* Input Section Box */
    .input-container {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 25px;
    }
    
    /* Global Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: #64748b;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MACHINE LEARNING ENGINE
# ==========================================

# Robust Dataset Loader
try:
    data = pd.read_csv("student_pass_fail.csv")
except FileNotFoundError:
    # Fallback synthetic dataset to keep engine running seamlessly
    np.random.seed(42)
    data = pd.DataFrame({
        "Study_Hours": np.random.randint(1, 11, 200),
        "Attendance": np.random.randint(40, 101, 200),
        "Assignment_Score": np.random.randint(20, 101, 200),
        "Sleep_Hours": np.random.randint(4, 10, 200),
        "Test_Score": np.random.randint(20, 101, 200),
        "Result": np.random.choice([0, 1], 200)
    })

# Define Model Parameters
X = data.drop(columns=["Result"])
y = data["Result"]
feature_names = list(X.columns)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Classifier
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Performance Validation
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


# ==========================================
# MAIN INTERFACE DESIGN
# ==========================================

# Application Header
st.title("🎓 Smart Student Performance Predictor")
st.markdown("An advanced predictive analytics system built by **Easy Business Technology** to accurately forecast academic outcomes.")
st.markdown("---")

# INTERACTIVE PREDICTION ZONE (Moved out of Sidebar into Main View)
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.subheader("📥 Interactive Student Data Form")
st.markdown("Adjust the variables below to recalculate the student's passing probability instantly:")

# Splitting inputs into responsive columns
form_col1, form_col2, form_col3 = st.columns(3)

with form_col1:
    study_hours = st.slider("📚 Daily Study Hours", min_value=1, max_value=12, value=6)
    attendance = st.slider("🎯 Attendance Rate (%)", min_value=0, max_value=100, value=75)

with form_col2:
    assignment_score = st.slider("📝 Assignment Score (0-100)", min_value=0, max_value=100, value=65)
    sleep_hours = st.slider("😴 Daily Sleep Hours", min_value=3, max_value=10, value=7)

with form_col3:
    test_score = st.slider("⚡ Test Score (0-100)", min_value=0, max_value=100, value=60)

st.markdown('</div>', unsafe_allow_html=True)

# Build Operational Array with Exact Column Alignment
live_input_data = pd.DataFrame(
    [[study_hours, attendance, assignment_score, sleep_hours, test_score]], 
    columns=feature_names
)
live_scaled_data = scaler.transform(live_input_data)

# Run Inference
live_prediction = model.predict(live_scaled_data)
live_probability = model.predict_proba(live_scaled_data)[0][1]


# ==========================================
# LIVE RESULT PRESENTATION
# ==========================================
st.subheader("🎯 Prediction Output")
res_col1, res_col2 = st.columns([2, 3])

with res_col1:
    if live_prediction[0] == 1:
        st.markdown(f"""
        <div style="background-color: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 25px; border-radius: 12px; margin-top: 15px;">
            <h2 style="color: #10b981; margin: 0;">🎉 Student  Pass</h2>
            <p style="color: #a7f3d0; margin: 10px 0 0 0;">Metrics indicate excellent performance and high consistency!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; padding: 25px; border-radius: 12px; margin-top: 15px;">
            <h2 style="color: #ef4444; margin: 0;">❌ Student  Failed</h2>
            <p style="color: #fca5a5; margin: 10px 0 0 0;">Warning: Risk of academic failure. Student requires immediate mentorship!</p>
        </div>
        """, unsafe_allow_html=True)

with res_col2:
    # Probability Speedometer gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=live_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Success Probability Rate", 'font': {'color': "#ffffff", 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ]
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=160, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})


# ==========================================
# DATA VISUALIZATION & ANALYTICS
# ==========================================
st.markdown("### 📈 Machine Learning Validation Status")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Overall Accuracy", f"{round(accuracy*100, 1)}%")
metric_col2.metric("Precision Score", f"{round(precision*100, 1)}%")
metric_col3.metric("Recall Metric", f"{round(recall*100, 1)}%")
metric_col4.metric("F1 Performance", f"{round(f1*100, 1)}%")

st.markdown("---")
st.subheader("📊 Historical Data Visual Analytics")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_pie = px.pie(
        data, 
        names="Result", 
        title="Dataset Balance: Pass vs Fail Distribution",
        hole=0.6,
        color_discrete_sequence=["#ef4444", "#10b981"],
        template="plotly_dark"
    )
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    fig_hist = px.histogram(
        data, 
        x="Study_Hours", 
        color="Result", 
        title="Impact Assessment: Daily Study Hours vs Outcome",
        barmode="group",
        color_discrete_sequence=["#ef4444", "#10b981"],
        template="plotly_dark"
    )
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

# Scatter Chart
fig_scatter = px.scatter(
    data, 
    x="Attendance", 
    y=feature_names[4], # Corresponds to Test Score column dynamically
    color="Result", 
    size="Assignment_Score",
    title="Comprehensive Distribution: Attendance vs Test Scores",
    color_discrete_sequence=["#ef4444", "#10b981"],
    template="plotly_dark"
)
fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_scatter, use_container_width=True)


# ==========================================
# CONFUSION MATRIX INTERPRETATION
# ==========================================
st.markdown("---")
st.subheader("📌 Model Diagnostics: Confusion Matrix")
matrix_col1, matrix_col2 = st.columns([2, 2])

with matrix_col1:
    cm = confusion_matrix(y_test, predictions)
    cm_dataframe = pd.DataFrame(cm, index=["Actual Fail", "Actual Pass"], columns=["Predicted Fail", "Predicted Pass"])
    st.dataframe(cm_dataframe, use_container_width=True)

with matrix_col2:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        <h4 style="margin-top:0; color:#38bdf8;">💡 Optimization Guidelines:</h4>
        <ul>
            <li><b>True Positives / Negatives:</b> Higher values represent absolute accuracy in AI predictions.</li>
            <li><b>False Positives / Negatives:</b> Lower values signify reduced algorithmic mistakes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Application Footer
st.markdown('<div class="footer">🚀 System Designed by Easy Business Technology | Powered by @updatecodesml</div>', unsafe_allow_html=True)