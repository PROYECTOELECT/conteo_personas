import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'clave_secreta_local_2026_conteo_personas')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# En Vercel el sistema de archivos es de solo lectura excepto /tmp
IS_VERCEL = bool(os.environ.get('VERCEL') or os.environ.get('VERCEL_ENV'))
BASE_DIR = '/tmp' if IS_VERCEL else os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['DATABASE'] = os.path.join(BASE_DIR, 'people_count.db')

# Ruta del modelo (siempre relativa al proyecto)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov5n.onnx')
PERSON_CLASS_ID = 0

# Carga perezosa del modelo (evita fallos al arrancar en Vercel)
_net = None

def get_net():
    global _net
    if _net is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Modelo no encontrado: {MODEL_PATH}')
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return _net

def get_db():
    os.makedirs(os.path.dirname(app.config['DATABASE']) or '.', exist_ok=True)
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        conn = get_db()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('master', 'admin', 'user')),
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                count INTEGER NOT NULL,
                photo_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        master = conn.execute("SELECT id FROM users WHERE role='master'").fetchone()
        if not master:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ('master', generate_password_hash('master123'), 'master')
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'init_db warning: {e}')

# Inicializar DB de forma segura
init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'role' not in session or session['role'] not in roles:
                flash('No tienes permisos para acceder a esta página.', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def detect_people(image_bytes):
    """Detecta solo personas humanas reales.
    Filtros estrictos: alta confianza + tamaño mínimo + proporción de cuerpo humano.
    Retorna (count, centers, boxes) donde boxes = [[x, y, w, h, conf], ...]
    conf = confianza 0-1 para mostrar el % en el recuadro verde.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 0, [], []
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net = get_net()
    net.setInput(blob)
    outputs = net.forward()

    boxes = []
    confidences = []
    centers = []

    CONF_THRESHOLD = 0.70
    MIN_HEIGHT_RATIO = 0.06
    MIN_WIDTH_RATIO = 0.02
    MIN_AREA_RATIO = 0.002
    MIN_ASPECT = 0.25
    MAX_ASPECT = 1.1

    for detection in outputs[0]:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if class_id != PERSON_CLASS_ID or confidence < CONF_THRESHOLD:
            continue

        cx = detection[0] * w
        cy = detection[1] * h
        bw = detection[2] * w
        bh = detection[3] * h

        if bh <= 0 or bw <= 0:
            continue
        if bh < h * MIN_HEIGHT_RATIO:
            continue
        if bw < w * MIN_WIDTH_RATIO:
            continue
        if bw * bh < (w * h * MIN_AREA_RATIO):
            continue
        aspect = bw / bh
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        left = int(cx - bw / 2)
        top = int(cy - bh / 2)
        boxes.append([left, top, int(bw), int(bh)])
        confidences.append(confidence)
        centers.append([int(cx), int(cy)])

    if not boxes:
        return 0, [], []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.4)
    if len(indices) == 0:
        return 0, [], []

    final_centers = []
    final_boxes = []
    for i in indices.flatten():
        i = int(i)
        cx, cy = int(centers[i][0]), int(centers[i][1])
        x, y, bw, bh = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        conf = float(round(confidences[i], 2))
        final_centers.append([cx, cy])
        # [x, y, w, h, confidence] — tipos nativos JSON
        final_boxes.append([x, y, bw, bh, conf])
    return len(final_centers), final_centers, final_boxes

def count_people(image_bytes):
    count, _, _ = detect_people(image_bytes)
    return count

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'Bienvenido, {username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Usuario o contraseña incorrectos.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Sesión cerrada.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')

@app.route('/count', methods=['POST'])
@login_required
def count_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    image_bytes = file.read()
    people_count, centers, boxes = detect_people(image_bytes)
    
    save = request.form.get('save', 'true').lower() != 'false'
    photo_path = None
    if save:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['user_id']}.jpg"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(photo_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save to DB only if saving
        conn = get_db()
        conn.execute(
            'INSERT INTO counts (user_id, count, photo_path) VALUES (?, ?, ?)',
            (session['user_id'], people_count, photo_path)
        )
        conn.commit()
        conn.close()
    
    return jsonify({
        'count': people_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'centers': centers,
        'boxes': boxes,
        'username': session.get('username', ''),
        'saved': save
    })

@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html')

@app.route('/api/reports')
@login_required
def api_reports():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    conn = get_db()
    query = '''
        SELECT c.id, c.timestamp, c.count, u.username
        FROM counts c
        JOIN users u ON c.user_id = u.id
        WHERE 1=1
    '''
    params = []
    if session['role'] == 'user':
        query += ' AND c.user_id = ?'
        params.append(session['user_id'])
    if start_date:
        query += ' AND DATE(c.timestamp) >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND DATE(c.timestamp) <= ?'
        params.append(end_date)
    query += ' ORDER BY c.timestamp DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    data = [dict(row) for row in rows]
    return jsonify(data)

@app.route('/export/excel')
@login_required
def export_excel():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    conn = get_db()
    query = '''
        SELECT c.timestamp, c.count, u.username
        FROM counts c JOIN users u ON c.user_id = u.id WHERE 1=1
    '''
    params = []
    if session['role'] == 'user':
        query += ' AND c.user_id = ?'
        params.append(session['user_id'])
    if start_date:
        query += ' AND DATE(c.timestamp) >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND DATE(c.timestamp) <= ?'
        params.append(end_date)
    query += ' ORDER BY c.timestamp DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reporte Conteo Personas"
    ws.append(['Fecha y Hora', 'Cantidad de Personas', 'Usuario'])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    for row in rows:
        ws.append([row['timestamp'], row['count'], row['username']])
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        ws.column_dimensions[column].width = max_length + 2
    
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    filename = f"reporte_conteo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/export/pdf')
@login_required
def export_pdf():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    conn = get_db()
    query = '''
        SELECT c.timestamp, c.count, u.username
        FROM counts c JOIN users u ON c.user_id = u.id WHERE 1=1
    '''
    params = []
    if session['role'] == 'user':
        query += ' AND c.user_id = ?'
        params.append(session['user_id'])
    if start_date:
        query += ' AND DATE(c.timestamp) >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND DATE(c.timestamp) <= ?'
        params.append(end_date)
    query += ' ORDER BY c.timestamp DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Reporte de Conteo de Personas", styles['Title']))
    elements.append(Spacer(1, 12))
    if start_date or end_date:
        rango = f"Desde: {start_date or 'inicio'} - Hasta: {end_date or 'fin'}"
        elements.append(Paragraph(rango, styles['Normal']))
        elements.append(Spacer(1, 12))
    
    data = [['Fecha y Hora', 'Cantidad', 'Usuario']]
    for row in rows:
        data.append([row['timestamp'], str(row['count']), row['username']])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    filename = f"reporte_conteo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, download_name=filename, as_attachment=True, mimetype='application/pdf')

@app.route('/users')
@login_required
@role_required(['master', 'admin'])
def users():
    conn = get_db()
    if session['role'] == 'master':
        users_list = conn.execute('SELECT id, username, role, created_at FROM users ORDER BY id').fetchall()
    else:  # admin
        users_list = conn.execute(
            "SELECT id, username, role, created_at FROM users WHERE role='user' OR id=?",
            (session['user_id'],)
        ).fetchall()
    conn.close()
    return render_template('users.html', users=users_list)

@app.route('/users/create', methods=['POST'])
@login_required
@role_required(['master', 'admin'])
def create_user():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    if session['role'] == 'admin' and role != 'user':
        flash('Los administradores solo pueden crear usuarios generales.', 'error')
        return redirect(url_for('users'))
    if not username or not password:
        flash('Usuario y contraseña requeridos.', 'error')
        return redirect(url_for('users'))
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO users (username, password_hash, role, created_by) VALUES (?, ?, ?, ?)',
            (username, generate_password_hash(password), role, session['user_id'])
        )
        conn.commit()
        flash(f'Usuario {username} creado correctamente.', 'success')
    except sqlite3.IntegrityError:
        flash('El nombre de usuario ya existe.', 'error')
    conn.close()
    return redirect(url_for('users'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
