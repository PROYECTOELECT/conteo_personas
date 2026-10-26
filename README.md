# Sistema de Conteo de Personas por Cámara

Aplicación web local para contar personas usando la cámara del dispositivo (celular, tablet o computador). Funciona en espacios abiertos o cerrados.

## Características
- Login con roles: **master**, **admin** y **usuario general**.
- Conteo por foto capturada, **subida de imagen** o en tiempo real.
- En tiempo real y en imágenes se dibujan **recuadros verdes** (bounding boxes) sobre cada persona + contador en pantalla.
- **Ubicación GPS en tiempo real** (lat/lng, altura, barrio, ciudad, país, continente y dirección precisa) usando el GPS del dispositivo. Aparece en pantalla y en la imagen exportada.
- Al finalizar aparece **Completado** y opción de exportar la imagen anotada con datos técnicos + ubicación.
- Reportes filtrables por rango de fechas.
- Exportación a Excel y PDF.
- Interfaz responsive (web y móvil).

## Usuario inicial
- Usuario: `master`
- Contraseña: `master123`

El usuario master puede crear administradores y usuarios generales. Los administradores pueden crear usuarios generales.

## Cómo ejecutar (local)

1. Asegúrate de tener Python 3.10+ y las dependencias instaladas:
   ```bash
   pip install flask opencv-python-headless numpy openpyxl reportlab
   ```

2. En la carpeta del proyecto:
   ```bash
   python3 app.py
   ```

3. Abre en el navegador:
   - En la misma máquina: http://127.0.0.1:5000
   - Desde otro dispositivo en la misma red: http://IP_DE_LA_MAQUINA:5000

4. Permite el acceso a la cámara cuando se solicite.

## Despliegue en GitHub + Vercel (para usar desde el móvil)

### 1. Subir a GitHub
1. Crea un repositorio nuevo en GitHub (público o privado).
2. En tu computadora, dentro de la carpeta del proyecto:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Conteo de personas"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

### 2. Desplegar en Vercel
1. Ve a [vercel.com](https://vercel.com) e inicia sesión con GitHub.
2. Haz clic en **Add New → Project** e importa el repositorio.
3. Vercel detectará Python. Confirma:
   - Framework Preset: Other
   - Root Directory: `./`
   - Build Command: (dejar vacío o `pip install -r requirements.txt`)
4. Haz clic en **Deploy**.

### Notas importantes para Vercel
- Vercel es serverless. La base de datos SQLite y la carpeta `uploads/` son **efímeras** (se pierden entre invocaciones). Para producción real se recomienda usar una base de datos externa (PostgreSQL, Supabase, etc.) y almacenamiento de archivos (S3, Cloudinary).
- El modelo YOLO se incluye en el repositorio (`models/yolov5n.onnx`).
- El acceso a cámara y GPS funciona desde el navegador del celular (debe ser HTTPS, que Vercel proporciona).
- Si el despliegue falla por tamaño o tiempo de build, considera usar Railway, Render o un VPS.

### Usuario inicial en producción
- Usuario: `master`
- Contraseña: `master123`  
  (cámbiala después del primer login creando un nuevo usuario master).

## Estructura
- `app.py` — aplicación principal
- `templates/` — interfaces HTML
- `models/` — modelo de detección
- `requirements.txt` — dependencias
- `vercel.json` — configuración de despliegue
- `uploads/` — fotos capturadas (local)

