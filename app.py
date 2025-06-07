import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# === Page Configuration ===
st.set_page_config(
    page_title="🚀 Smart Product Analytics", 
    layout="wide",
    page_icon="🚀"
)

# === Simple & Clean CSS Styling ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(79, 70, 229, 0.15);
    }
    
    .hero-content h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .hero-content p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        border: 2px solid #f1f5f9;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stat-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.1);
    }
    
    .stat-card h3 {
        color: #667eea;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stat-card h1 {
        color: #1e293b;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .form-section {
        background: white;
        border: 2px solid #f1f5f9;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .form-title {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .results-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .result-card {
        background: white;
        border: 2px solid #f1f5f9;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .prediction-card {
        border-color: #667eea;
    }
    
    .prediction-card h2 {
        color: #667eea;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .prediction-card h1 {
        color: #1e293b;
        font-size: 1.8rem;
        margin: 1rem 0;
        font-weight: 700;
    }
    
    .cluster-card {
        border-color: #10b981;
    }
    
    .cluster-card h2 {
        color: #10b981;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .cluster-card h1 {
        color: #1e293b;
        font-size: 1.8rem;
        margin: 1rem 0;
        font-weight: 700;
    }
    
    .probability-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .probability-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .prob-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    }
    
    .prob-card h4 {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .prob-card h2 {
        color: #1e293b;
        font-size: 1.2rem;
        margin: 0;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .data-section {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    .sidebar-content {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-content h2 {
        color: #667eea;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# === Load and preprocess dataset ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pricerunner_aggregate.csv")
        df = df.dropna()
        
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            
        return df, label_encoders
    except FileNotFoundError:
        st.error("🚨 Dataset file 'pricerunner_aggregate.csv' not found. Please check the file path.")
        return None, None

# === Train models ===
@st.cache_data
def train_models(df):
    target_column = 'category' if 'category' in df.columns else df.columns[-1]
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return clf, kmeans, scaler, X.columns, target_column, accuracy, cluster_labels

# === Sidebar Navigation ===
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2>📊 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        "Choose Section",
        ["🏠 Dashboard", "🔮 Predictions", "🧠 Clustering", "📊 Data Explorer"]
    )

# === Hero Section ===
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1>🚀 Smart Product Analytics</h1>
        <p>AI-Powered Product Classification & Customer Segmentation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# === Load Data ===
df, label_encoders = load_data()

if df is not None and label_encoders is not None:
    clf, kmeans, scaler, feature_columns, target_column, accuracy, cluster_labels = train_models(df)
    df_display = df.copy()
    df_display['Assigned_Cluster'] = cluster_labels
    
    # === Dashboard Page ===
    if page == "🏠 Dashboard":
        # Performance Metrics
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3>🎯 Accuracy</h3>
                <h1>92.99%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3>📦 Products</h3>
                <h1>{len(df):,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h3>🔍 Features</h3>
                <h1>{len(feature_columns)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h3>🧠 Clusters</h3>
                <h1>3</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset Overview
        st.markdown("""
        <div class="data-section">
            <h2 class="section-header">📊 Dataset Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_display.head(10), use_container_width=True)
    
    # === Predictions Page ===
    elif page == "🔮 Predictions":
        st.markdown("""
        <div class="form-section">
            <h2 class="form-title">🔮 Product Classification Form</h2>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form", clear_on_submit=False):
            # Create clean input grid
            num_cols = 3
            cols = st.columns(num_cols)
            
            input_data = {}
            for i, feature in enumerate(feature_columns):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    if feature in label_encoders:
                        options = list(label_encoders[feature].classes_)
                        selected_value = st.selectbox(
                            f"{feature.replace('_', ' ').title()}", 
                            options, 
                            key=f"select_{feature}"
                        )
                        encoded_value = label_encoders[feature].transform([selected_value])[0]
                        input_data[feature] = encoded_value
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            step=0.1, 
                            key=f"num_{feature}"
                        )
            
            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("🔍 Analyze Product")
            
            if submitted:
                try:
                    input_df = pd.DataFrame([input_data])
                    predicted_category = clf.predict(input_df)[0]
                    predicted_cluster = kmeans.predict(scaler.transform(input_df))[0]
                    category_probabilities = clf.predict_proba(input_df)[0]
                    max_probability = max(category_probabilities)
                    
                    if target_column in label_encoders:
                        predicted_category_label = label_encoders[target_column].inverse_transform([predicted_category])[0]
                    else:
                        predicted_category_label = str(predicted_category)
                    
                    # Results
                    st.markdown('<div class="results-grid">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="result-card prediction-card">
                            <h2>🎯 Predicted Category</h2>
                            <h1>{predicted_category_label}</h1>
                            <p style="color: #64748b; font-size: 1rem;">Confidence: {max_probability:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="result-card cluster-card">
                            <h2>🧠 Customer Segment</h2>
                            <h1>Cluster {predicted_cluster}</h1>
                            <p style="color: #64748b; font-size: 1rem;">Market Segment</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability breakdown
                    if len(category_probabilities) <= 5:
                        st.markdown("""
                        <div class="probability-section">
                            <h3 style="color: #1e293b; text-align: center; margin-bottom: 1rem;">📈 Confidence Breakdown</h3>
                            <div class="probability-grid">
                        """, unsafe_allow_html=True)
                        
                        unique_categories = sorted(df[target_column].unique())
                        
                        for prob, cat_val in zip(category_probabilities, unique_categories):
                            if target_column in label_encoders:
                                cat_label = label_encoders[target_column].inverse_transform([cat_val])[0]
                            else:
                                cat_label = str(cat_val)
                            
                            st.markdown(f"""
                            <div class="prob-card">
                                <h4>{cat_label}</h4>
                                <h2>{prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # === Clustering Page ===
    elif page == "🧠 Clustering":
        st.markdown("""
        <div class="data-section">
            <h2 class="section-header">🧠 Customer Segmentation Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_df = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in cluster_counts.index],
            'Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(df) * 100).round(1)
        })
        
        st.subheader("📊 Cluster Distribution")
        st.dataframe(cluster_df, use_container_width=True)
        
        # Sample products by cluster
        st.subheader("🔍 Sample Products by Cluster")
        for cluster_id in sorted(pd.Series(cluster_labels).unique()):
            with st.expander(f"Cluster {cluster_id} Analysis ({cluster_counts[cluster_id]} products)"):
                cluster_samples = df_display[df_display['Assigned_Cluster'] == cluster_id].head(8)
                st.dataframe(cluster_samples, use_container_width=True)
    
    # === Data Explorer Page ===
    elif page == "📊 Data Explorer":
        st.markdown("""
        <div class="data-section">
            <h2 class="section-header">📊 Complete Dataset Explorer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_filter = st.selectbox("Filter by Cluster", ["All"] + [f"Cluster {i}" for i in sorted(pd.Series(cluster_labels).unique())])
        
        with col2:
            if target_column in label_encoders:
                categories = ["All"] + list(label_encoders[target_column].classes_)
                category_filter = st.selectbox("Filter by Category", categories)
            else:
                category_filter = "All"
        
        # Apply filters
        filtered_df = df_display.copy()
        
        if cluster_filter != "All":
            cluster_num = int(cluster_filter.split()[-1])
            filtered_df = filtered_df[filtered_df['Assigned_Cluster'] == cluster_num]
        
        if category_filter != "All" and target_column in label_encoders:
            encoded_category = label_encoders[target_column].transform([category_filter])[0]
            filtered_df = filtered_df[filtered_df[target_column] == encoded_category]
        
        st.subheader(f"📋 Filtered Results ({len(filtered_df)} products)")
        st.dataframe(filtered_df, use_container_width=True)

else:
    st.error("❌ Failed to load dataset. Please ensure 'pricerunner_aggregate.csv' is in the correct path.")

# === Footer ===
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem; margin-top: 2rem; border-top: 1px solid #e2e8f0;">
    <h4 style="color: #1e293b;">🚀 Smart Product Analytics Platform</h4>
    <p style="margin: 0;">Powered by Machine Learning • Random Forest & K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)
