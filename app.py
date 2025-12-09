# app.py
import os
import io
import uuid
import json
import base64
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, send_file)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
import pyrebase
from firebase_admin import credentials, firestore
import firebase_admin
from dotenv import load_dotenv
import plotly.io as pio



# Enable Kaleido (required for PNG export on Render)
pio.io.kaleido.enabled = True

# Global PNG export settings
pio.defaults.width = 900
pio.defaults.height = 600
pio.defaults.scale = 1

# Chromium path used by Render
pio.defaults.chromium_path = "/usr/bin/chromium"


load_dotenv()

# ---------------- Flask app ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(24))

FILE_STORE = {}

# ---------------- Firebase Web Config (SAFE) ----------------
firebase_config_env = os.getenv("FIREBASE_WEB_CONFIG")
firebaseConfig = json.loads(firebase_config_env) if firebase_config_env else {}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# ---------------- Firebase Admin (SAFE) ----------------
firebase_admin_json = os.getenv("FIREBASE_ADMIN_JSON")

if firebase_admin_json:
    try:
        admin_dict = json.loads(firebase_admin_json)
        cred = credentials.Certificate(admin_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        print("Firebase admin init error:", e)
        db = None
else:
    db = None



# ---------------- Helpers ----------------
def safe_read_csv(text):
    encodings = ['utf-8', 'latin1', 'cp1252']
    delims = [None, ',', ';', '\t', '|']
    last_err = None
    for enc in encodings:
        try:
            decoded = text if isinstance(text, str) else text.decode(enc)
        except Exception as e:
            last_err = e
            continue
        for delim in delims:
            try:
                if delim:
                    df = pd.read_csv(io.StringIO(decoded), sep=delim)
                else:
                    df = pd.read_csv(io.StringIO(decoded))
                return df
            except Exception as e:
                last_err = e
                continue
    raise last_err or ValueError("Unable to parse CSV")

def sample_for_plot(df, n=1000):
    try:
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=42)
    except Exception:
        return df.head(n)

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            # avoid showing "Please log in" repeatedly across redirects
            if not request.endpoint == 'login':
                if 'flash_shown' not in session:
                    flash("Please log in.", "warning")
                    session['flash_shown'] = True
            return redirect(url_for('login'))
        # Reset flash flag after successful login
        session.pop('flash_shown', None)
        return f(*args, **kwargs)
    return wrapper

def get_store():
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
        FILE_STORE[sid] = {}
    if sid not in FILE_STORE:
        FILE_STORE[sid] = {}
    return FILE_STORE[sid]

# ---------------- PDF helper components ----------------
LOGO_PATH = os.path.join("static", "images", "datasci_logo.png")  # your chosen B filename

def safe_rl_image_from_b64(b64str, width=None, height=None):
    """Return a ReportLab Image object created from a base64 image string (or raise)."""
    img_bytes = base64.b64decode(b64str)
    img_buf = io.BytesIO(img_bytes)
    if width and height:
        return RLImage(img_buf, width=width, height=height)
    else:
        return RLImage(img_buf)

def build_header_table(styles, title_text="DataSci – AI Powered Data Analysis"):
    # Try to include the logo if available; if not, render title only
    logo_exists = os.path.exists(LOGO_PATH)
    logo_rl = None
    if logo_exists:
        try:
            logo_rl = RLImage(LOGO_PATH, width=40, height=40)
        except Exception:
            logo_rl = None

    # Build a simple table with logo left and title right on blue background
    if logo_rl:
        header = Table([[logo_rl, Paragraph(f"<b>{title_text}</b>", styles['Title'])]],
                       colWidths=[50, 440])
    else:
        header = Table([[Paragraph(f"<b>{title_text}</b>", styles['Title'])]],
                       colWidths=[490])

    header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0d6efd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return header

def build_footer_paragraph(styles):
    txt = ("<para align='center'>"
           "<b>Aluvala Ediga Harsha Vardhan Goud</b><br/>"
           "MCA | AI & ML Developer • GitHub: <u>github.com/Aluval</u><br/>"
           "© 2025 DataSci Platform"
           "</para>")
    return Paragraph(txt, styles['Normal'])

def make_bordered_box(flowables, width=500):
    """Wrap a list of flowables (or a single flowable) into a bordered Table box."""
    # If flowables is a list, put them inside a single cell vertically by concatenating with Spacer
    if isinstance(flowables, list):
        cell_content = flowables
    else:
        cell_content = [flowables]
    # We create a single-cell table and put the flowables in that cell
    t = Table([[cell_content]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    return t

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        fullName = request.form.get('fullName')
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        agree = request.form.get('agree')

        if not agree:
            flash("You must agree to the Terms & Privacy Policy.", "warning")
            return redirect(url_for('register'))

        try:
            user = auth.create_user_with_email_and_password(email, password)
            if db:
                db.collection('users').document(user['localId']).set({
                    'fullName': fullName, 'email': email, 'mobile': mobile
                })
            flash("Account created successfully — please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Registration failed: {e}", "danger")
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        agree = request.form.get('agree')

        if not agree:
            flash("Please agree to the Terms & Privacy Policy to continue.", "warning")
            return redirect(url_for('login'))

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
            session['user_id'] = user.get('localId')
            flash("Logged in.", "success")
            return redirect(url_for('upload'))
        except Exception as e:
            flash(f"Login failed: {e}", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    sid = session.get('sid')
    if sid in FILE_STORE:
        FILE_STORE.pop(sid, None)
    session.clear()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash("No file selected.", "warning")
            return redirect(request.url)
        raw = f.read()
        text = None
        for enc in ("utf-8","latin1","cp1252"):
            try:
                text = raw.decode(enc)
                break
            except:
                pass
        if text is None:
            flash("Unable to decode file.", "danger")
            return redirect(url_for('upload'))

        store = get_store()
        store['csv_text'] = text
        store['filename'] = f.filename
        for k in ['predicted_csv','last_visual_img','last_prediction_img','last_metrics','last_plot_html','preview_html','last_trend_html','last_donut_html']:
            store.pop(k, None)
        flash("File uploaded.", "success")
        return redirect(url_for('dashboard'))
    return render_template('upload.html')

@app.route('/dashboard')
@login_required
def dashboard():
    store = get_store()
    csv_text = store.get('csv_text')
    if not csv_text:
        flash("Upload CSV first.", "warning")
        return redirect(url_for('upload'))
    try:
        df = safe_read_csv(csv_text)
    except Exception as e:
        flash(f"Failed to parse CSV: {e}", "danger")
        return redirect(url_for('upload'))

    df.columns = [str(c).strip() for c in df.columns]
    numeric = df.select_dtypes(include='number')
    if numeric.empty:
        coerced = df.apply(lambda c: pd.to_numeric(c.astype(str).str.replace(',','').str.strip(), errors='coerce'))
        numeric = coerced.select_dtypes(include='number')

    kpis = []
    colors = ["#0d6efd","#0aa0df","#00b388","#f6c026","#e74c3c","#8e44ad"]
    for i, col in enumerate(numeric.columns[:6]):
        try:
            if any(w in col.lower() for w in ['revenue','profit','total','sales','amount','cost']):
                val = numeric[col].sum()
            else:
                val = numeric[col].mean()
            kpis.append({"name": col, "value": round(float(val),2), "color": colors[i % len(colors)]})
        except:
            kpis.append({"name": col, "value": "N/A", "color": colors[i % len(colors)]})

    trend_plot = ""
    donut_plot = ""
    if not numeric.empty:
        first = numeric.columns[0]
        try:
            plot_df = sample_for_plot(df[[first]].reset_index().rename(columns={'index':'Index'}))
            fig = px.line(plot_df, x='Index', y=first, title=f"{first} Trend")
            trend_plot = fig.to_html(full_html=False)
            store['last_trend_html'] = trend_plot
        except:
            pass
        try:
            if len(df) <= 50:
                fig2 = px.pie(df, names=df.index.astype(str), values=first, title=f"{first} Distribution")
            else:
                s = pd.to_numeric(df[first], errors='coerce').fillna(0)
                top_idx = s.nlargest(10).index
                fig2 = px.bar(x=[str(i) for i in top_idx], y=s.loc[top_idx].values, title=f"Top 10 {first} Values", labels={'x':'Index','y':first})
            donut_plot = fig2.to_html(full_html=False)
            store['last_donut_html'] = donut_plot
        except:
            pass

    preview_html = df.head(20).to_html(classes='table table-striped', index=False)
    store['preview_html'] = preview_html

    return render_template('dashboard.html',
                           kpis=kpis,
                           trend_plot=trend_plot,
                           donut_plot=donut_plot,
                           table_html=preview_html,
                           last_metrics=store.get('last_metrics', {}))

@app.route('/visualize', methods=['GET','POST'])
@login_required
def visualize():
    store = get_store()
    csv_text = store.get('csv_text')
    if not csv_text:
        flash("Upload CSV first.", "warning")
        return redirect(url_for('upload'))
    df = safe_read_csv(csv_text)
    columns = df.columns.tolist()
    plot_html = store.get('last_plot_html', '')
    preview_html = df.head(20).to_html(classes='table table-striped', index=False)

    if request.method == 'POST':
        x = request.form.get('x_col')
        y = request.form.get('y_col')
        plot_type = request.form.get('plot_type')
        try:
            plot_df = sample_for_plot(df)
            fig = None
            if plot_type == 'scatter' and x and y:
                fig = px.scatter(plot_df, x=x, y=y, title=f"{x} vs {y}")
            elif plot_type == 'line' and x and y:
                fig = px.line(plot_df, x=x, y=y, title=f"{x} vs {y}")
            elif plot_type == 'bar' and x and y:
                fig = px.bar(plot_df, x=x, y=y, title=f"{x} vs {y}")
            elif plot_type == 'histogram' and x:
                fig = px.histogram(plot_df, x=x, title=f"Histogram of {x}")
            elif plot_type == 'heatmap':
                fig = px.imshow(plot_df.corr(numeric_only=True), title='Correlation Heatmap')
            if fig is not None:
                plot_html = fig.to_html(full_html=False)
                store['last_plot_html'] = plot_html
                # try to save as base64 image for PDF
                try:
                    img_bytes = pio.to_image(fig, format='png')
                    store['last_visual_img'] = base64.b64encode(img_bytes).decode()
                except Exception as e:
                    # image creation may fail on some environments; remove key so PDF falls back gracefully
                    store.pop('last_visual_img', None)
        except Exception as e:
            flash(f"Plot error: {e}", "danger")

    return render_template('visualize.html', columns=columns, plot_html=plot_html, preview_html=preview_html)

@app.route('/visualize_print')
@login_required
def visualize_print():
    store = get_store()
    csv_text = store.get('csv_text') or ''
    df = safe_read_csv(csv_text) if csv_text else pd.DataFrame()

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    # Header
    content.append(build_header_table(styles))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Visualization Report</b>", styles['Heading2']))
    content.append(Spacer(1, 10))

    # Attempt to add image (if available)
    img_b64 = store.get('last_visual_img')
    if img_b64:
        try:
            rl_img = safe_rl_image_from_b64(img_b64, width=450, height=300)
            content.append(rl_img)
            content.append(Spacer(1, 12))
        except Exception:
            content.append(Paragraph("Visualization image not available for PDF (fallback to table).", styles['Normal']))
            content.append(Spacer(1, 8))
    else:
        content.append(Paragraph("No visualization image saved. See dashboard for interactive chart.", styles['Normal']))
        content.append(Spacer(1, 8))

    # Build table data safely
    try:
        head = list(df.head(10).columns)
        rows = df.head(10).values.tolist()
        # If no columns or rows, create a single "No data" table
        if len(head) == 0 or len(rows) == 0:
            table_data = [["No data available"]]
            col_count = 1
        else:
            table_data = [head] + rows
            col_count = len(head)
    except Exception:
        table_data = [["No data available"]]
        col_count = 1

    # Create table with style (handle single cell table)
    report_table = Table(table_data, hAlign='CENTER')
    report_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))

    # Wrap table in a bordered box for the visual effect
    boxed = make_bordered_box([report_table])
    content.append(boxed)
    content.append(Spacer(1, 20))

    # Footer
    content.append(build_footer_paragraph(styles))

    # Build doc
    doc.build(content)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name='visualize_report.pdf')

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    store = get_store()
    csv_text = store.get('csv_text')
    if not csv_text:
        flash("Upload CSV first.", "warning")
        return redirect(url_for('upload'))
    df = safe_read_csv(csv_text)
    columns = df.columns.tolist()
    results_html = None
    scatter_html = None

    if request.method == 'POST':
        target = request.form.get('target')
        features = request.form.getlist('features')
        model_type = request.form.get('model')
        use_split = request.form.get('use_split') == 'on'
        test_size = float(request.form.get('test_size') or 0.2)

        if not target or not features:
            flash("Select target & features.", "warning")
            return redirect(url_for('predict'))

        # coerce features/target
        X_raw = df[features].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(',','').str.strip(), errors='coerce'))
        if model_type in ["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier","KNNClassifier","SVMClassifier"]:
            y_raw = df[target].astype(str)
        else:
            y_raw = pd.to_numeric(df[target].astype(str).str.replace(',','').str.strip(), errors='coerce')

        combined = pd.concat([X_raw, y_raw.rename('__target__')], axis=1).dropna(subset=['__target__'])
        X_cleaned = combined[features].fillna(combined[features].mean())
        y_cleaned = combined['__target__']

        if X_cleaned.empty or y_cleaned.empty:
            flash("After cleaning, dataset is empty or target missing — choose another target/feature.", "danger")
            return redirect(url_for('predict'))

        if model_type in ["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier","KNNClassifier","SVMClassifier"]:
            le = LabelEncoder()
            y = le.fit_transform(y_cleaned)
            store['le_classes'] = list(le.classes_)
            is_classification = True
        else:
            y = y_cleaned.to_numpy()
            is_classification = False

        # splits
        if model_type not in ["KMeans3","KMeans5"]:
            if use_split:
                X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=test_size, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X_cleaned, X_cleaned, y, y
        else:
            X_train, X_test, y_train, y_test = X_cleaned, X_cleaned, y, y

        # instantiate model
        model = None
        try:
            if model_type == "LinearRegression":
                model = LinearRegression()
            elif model_type == "DecisionTreeRegressor":
                model = DecisionTreeRegressor(random_state=42)
            elif model_type == "RandomForestRegressor":
                model = RandomForestRegressor(random_state=42)
            elif model_type == "KNNRegressor":
                model = KNeighborsRegressor()
            elif model_type == "SVR":
                model = SVR()
            elif model_type == "LogisticRegression":
                model = LogisticRegression(max_iter=2000)
            elif model_type == "DecisionTreeClassifier":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "RandomForestClassifier":
                model = RandomForestClassifier(random_state=42)
            elif model_type == "KNNClassifier":
                model = KNeighborsClassifier()
            elif model_type == "SVMClassifier":
                model = SVC()
            elif model_type == "KMeans3":
                model = KMeans(n_clusters=3, random_state=42, n_init='auto')
            elif model_type == "KMeans5":
                model = KMeans(n_clusters=5, random_state=42, n_init='auto')
            else:
                flash("Model not supported.", "warning")
                return redirect(url_for('predict'))

            # train & predict
            if model_type not in ["KMeans3","KMeans5"]:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                X_all_s = scaler.transform(X_cleaned)

                model.fit(X_train_s, y_train)
                preds_all = model.predict(X_all_s)

                df['Prediction'] = np.nan
                df.loc[X_cleaned.index, 'Prediction'] = preds_all

                if is_classification:
                    y_test_pred = model.predict(X_test_s)
                    acc = accuracy_score(y_test, y_test_pred)
                    metrics = {"accuracy": round(float(acc),4)}
                else:
                    y_test_pred = model.predict(X_test_s)
                    try:
                        r2 = r2_score(y_test, y_test_pred)
                        metrics = {"r2": round(float(r2),4)}
                    except:
                        metrics = {"r2": "N/A"}
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cleaned)
                df['Cluster'] = np.nan
                df.loc[X_cleaned.index, 'Cluster'] = model.fit_predict(X_scaled)
                metrics = {"clusters": int(df['Cluster'].nunique())}

            # save results and metrics
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            store['predicted_csv'] = buf.getvalue()
            store['last_metrics'] = metrics

            # try saving scatter image for PDF
            results_html = df.head(20).to_html(classes='table table-striped', index=False)
            if model_type not in ["KMeans3","KMeans5"] and 'Prediction' in df.columns and not is_classification:
                try:
                    df_plot = df.dropna(subset=['Prediction', target]).copy()
                    if not df_plot.empty:
                        sample_plot_df = sample_for_plot(df_plot, n=1000)
                        fig = px.scatter(sample_plot_df, x=target, y='Prediction', title='Actual vs Predicted')
                        max_val = max(sample_plot_df[target].max(), sample_plot_df['Prediction'].max())
                        min_val = min(sample_plot_df[target].min(), sample_plot_df['Prediction'].min())
                        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                      line=dict(color="Red", width=1, dash="dash"))
                        scatter_html = fig.to_html(full_html=False)
                        try:
                            img_bytes = pio.to_image(fig, format='png')
                            store['last_prediction_img'] = base64.b64encode(img_bytes).decode()
                        except:
                            store.pop('last_prediction_img', None)
                except Exception:
                    scatter_html = None

            short_prompt = f"Model: {model_type}. Metrics: {metrics}."
            store['prediction_summary'] = short_prompt
            flash("Prediction completed.", "success")
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template('predict.html',
                           columns=columns,
                           results_table=results_html,
                           scatter_html=scatter_html,
                           metrics=store.get('last_metrics'))

@app.route('/predict_print')
@login_required
def predict_print():
    store = get_store()
    pred_csv = store.get('predicted_csv') or store.get('csv_text', '')
    try:
        df = safe_read_csv(pred_csv)
    except Exception:
        df = pd.DataFrame()

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    # Header
    content.append(build_header_table(styles, title_text="DataSci – AI Powered Data Analysis"))
    content.append(Spacer(1, 12))

    # Title & metrics
    content.append(Paragraph("<b>Prediction Report</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))
    metrics = store.get("last_metrics", {})
    if metrics:
        content.append(Paragraph(f"<b>Metrics:</b> {metrics}", styles["Normal"]))
        content.append(Spacer(1, 8))

    # Prediction image (if available)
    img_b64 = store.get('last_prediction_img')
    if img_b64:
        try:
            img_rl = safe_rl_image_from_b64(img_b64, width=450, height=300)
            content.append(img_rl)
            content.append(Spacer(1, 12))
        except Exception:
            content.append(Paragraph("Prediction image not available for PDF.", styles["Normal"]))
            content.append(Spacer(1, 8))
    else:
        content.append(Paragraph("No prediction image saved. See results on the Predict page.", styles["Normal"]))
        content.append(Spacer(1, 8))

    # Table (safe handling)
    try:
        head = list(df.head(10).columns)
        rows = df.head(10).values.tolist()
        if len(head) == 0 or len(rows) == 0:
            table_data = [["No data available"]]
        else:
            table_data = [head] + rows
    except Exception:
        table_data = [["No data available"]]

    report_table = Table(table_data, hAlign='CENTER')
    report_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))

    content.append(make_bordered_box([report_table]))
    content.append(Spacer(1, 20))

    content.append(build_footer_paragraph(styles))
    doc.build(content)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name='prediction_report.pdf')

@app.route('/download_predicted')
@login_required
def download_predicted():
    store = get_store()
    csv_text = store.get('predicted_csv')
    if not csv_text:
        flash("No predicted CSV.", "warning")
        return redirect(url_for('predict'))
    return send_file(io.BytesIO(csv_text.encode()), mimetype="text/csv", as_attachment=True, download_name="predicted.csv")

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if not db:
        user = {"fullName": "", "email": session.get("user",""), "mobile": ""}
        if request.method == 'POST':
            flash("Profile storage not configured.", "warning")
            return redirect(url_for('profile'))
        return render_template('profile.html', user=user)
    user_id = session.get('user_id')
    if not user_id:
        flash("User ID missing", "warning")
        return redirect(url_for('login'))
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    user = doc.to_dict() if doc.exists else {"fullName":"","email":session.get("user",""),"mobile":""}
    if request.method == 'POST':
        fullName = request.form.get('fullName')
        mobile = request.form.get('mobile')
        try:
            doc_ref.update({'fullName': fullName, 'mobile': mobile})
            flash("Profile updated.", "success")
            return redirect(url_for('profile'))
        except Exception as e:
            flash(f"Update failed: {e}", "danger")
            return redirect(url_for('profile'))
    return render_template('profile.html', user=user)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, port=port)


