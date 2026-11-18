import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# PARÁMETROS EDITABLES
# =============================================================================
CARPETA = Path(__file__).parent
MODELO = CARPETA / "face_landmarker.task"
CARPETA_FOTOS = CARPETA / "fotos"

CAMARA_INDEX = 0
USAR_DSHOW = True        # si la imagen sale rara o con ruido, probá False
ANCHO_CAPTURA = 960
ALTO_CAPTURA = 720
MAX_CARAS = 2            # cuántas caras detecta a la vez

OPACIDAD = 0.85          # 1.0 = máscara opaca | 0.6 = se mezcla con tu cara
SUAVIZADO_BORDE = 0.22   # difuminado del borde (más alto = más suave)
IGUALAR_LUZ = 0.85       # 0 = no tocar la luz de la foto | 1 = igualarla a tu cara

# "color"    -> todo a color
# "bn_cara"  -> solo la máscara en blanco y negro
# "bn_toda"  -> toda la imagen en blanco y negro (recomendado)
MODO_COLOR = "bn_toda"
CONTRASTE_BN = 1.1       # 1.0 normal | más alto = más contraste
CAMBIAR_CADA_SEG = 8     # cada cuántos segundos rotan las fotos

LADO_MAX_FOTO = 800      # las fotos más grandes se reducen a este tamaño
PAD = 40                 # margen de seguridad alrededor de cada foto
TAM_TEXTO = 20
TEXTO_FIJO = ""          # texto fijo al pie de la pantalla (opcional)

# --- Panel de información al costado de la cara ---
TAM_INFO = 16            # tamaño de letra del panel
ANCHO_INFO = 300         # ancho del panel en píxeles
INFO_RETARDO = 1.5       # segundos desde que cambia la foto hasta que empieza a aparecer
INFO_FADE = 1.0          # duración del fundido de entrada
INFO_SEPARACION = 20     # distancia entre la cara y el panel

NOMBRE_VENTANA = "Obra"

# =============================================================================
# TEXTOS DE CADA FOTO
# La clave es el nombre del archivo sin extensión: "1" -> 1.jpg, "10" -> 10.jpg.
# Los saltos de línea (\n) separan párrafos.
# =============================================================================
TEXTOS = {
    "1": (
        "Agostina Vega, la adolescente de 14 años asesinada en mayo de 2026 "
        "en la provincia de Córdoba.\n"
        "Principal acusado: Claudio Gabriel Barrelier (33 años, expareja de la "
        "madre de la víctima). Llegará al banquillo acusado de abuso sexual con "
        "acceso carnal y homicidio triplemente calificado por alevosía, "
        "criminis causa y violencia de género (femicidio), además de privación "
        "ilegítima de la libertad en otra causa conexa."
    ),
    "2": (
        "Ángeles Rawson tenía 16 años. Ingresó a su edificio en la calle "
        "Ravignani 2360 el 10 de junio de 2013 por la mañana, y fue asesinada "
        "durante un intento de violación.\n"
        "El femicida: Jorge Néstor Mangeri."
    ),
    "3": (
        "El femicidio de Catalina Gutiérrez, ocurrido el 17 de julio de 2024 "
        "en la ciudad de Córdoba, fue el asesinato de una estudiante de "
        "Arquitectura e influencer de 21 años a manos de su amigo y compañero "
        "de facultad, Néstor Alejandro Aguilar Soto."
    ),
    "4": (
        "Rocío Alvarito fue una joven de 26 años que falleció el 29 de enero "
        "de 2026 tras caer desde el balcón de un segundo piso en la ciudad de "
        "La Plata.\n"
        "Por el hecho se encuentra detenido con prisión preventiva su entonces "
        "pareja, Marcos Ariel García, cuya causa ya cuenta con un pedido de "
        "elevación a juicio oral."
    ),
    "5": (
        "Micaela García tenía 21 años. Desapareció la madrugada del 1 de abril "
        "de 2017 al salir de un boliche en Gualeguay, Entre Ríos, y fue "
        "asesinada tras sufrir abuso sexual.\n"
        "El femicida: Sebastián Wagner."
    ),
    "6": (
        "Florencia Romano tenía 14 años. Acudió a la casa de un hombre que la "
        "había contactado por Instagram en Maipú, Mendoza, el 12 de diciembre "
        "de 2020, y fue golpeada y asesinada.\n"
        "El femicida: Pablo Arancibia."
    ),
    "7": (
        "Candela Sol Rodríguez tenía 11 años. Fue secuestrada el 22 de agosto "
        "de 2011 cerca de su casa en Villa Tesei, Hurlingham, y hallada "
        "asesinada por asfixia tras sufrir abuso sexual nueve días después.\n"
        "Los condenados a perpetua: Hugo Bermúdez y Leonardo Jara."
    ),
    "8": (
        "Cintia Díaz Orellana tenía 20 años. Fue atacada brutalmente a golpes "
        "en la vía pública tras salir de un boliche en San José de Metán, "
        "Salta, el 22 de agosto de 2026, y falleció al día siguiente por las "
        "heridas.\n"
        "El femicida imputado: Gastón Ramírez."
    ),
    "10": (
        "Johana Ramallo tenía 23 años. Desapareció el 26 de julio de 2017 en "
        "la zona roja de La Plata tras ser captada por una red de trata y "
        "explotación sexual, y restos de su cuerpo fueron hallados un año "
        "después en la costa de Berisso.\n"
        "El femicida: aún sin identificar (hay ocho imputados juzgados por "
        "trata y narcotráfico, encabezados por Carlos \"El Cabezón\" Rodríguez)."
    ),
    "11": (
        "Dulce María Beatriz Candia tenía 17 años. Desapareció el 17 de mayo "
        "de 2026 en Eldorado, Misiones, y su cuerpo fue hallado once días "
        "después en una obra en construcción abandonada tras haber sido "
        "asesinada por asfixia mecánica.\n"
        "El femicida imputado: Mario Alberto Y. (un remisero de 46 años)."
    ),
}

# =============================================================================
# PUNTOS DE LA CARA QUE SE USAN (índices de los 468 del Face Landmarker)
# =============================================================================
OVALO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
         379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
         234, 127, 162, 21, 54, 103, 67, 109]
OJO_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OJO_B = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CEJAS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
         300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NARIZ = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358]
MEJILLAS = [50, 280, 205, 425, 116, 345, 123, 352, 213, 433]
LABIOS_EXT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
              269, 267, 0, 37, 39, 40, 185]
LABIOS_INT = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310,
              311, 312, 13, 82, 81, 80, 191]

IDX_MALLA = list(dict.fromkeys(
    OVALO + OJO_A + OJO_B + CEJAS + NARIZ + MEJILLAS + LABIOS_EXT + LABIOS_INT
))


# =============================================================================
# TEXTOS
# =============================================================================
def cargar_fuente(tam):
    for ruta in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(ruta, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def normalizar(s):
    """Minúsculas y sin tildes, para comparar nombres de archivo."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def buscar_texto(nombre_archivo):
    nombre = normalizar(nombre_archivo)
    if nombre in TEXTOS:                      # coincidencia exacta (1, 2, 10...)
        return TEXTOS[nombre]
    for clave, texto in TEXTOS.items():       # por palabra dentro del nombre
        if not clave.isdigit() and clave in nombre:
            return texto
    return None


def crear_cartel(texto, fuente):
    """Genera una imagen BGRA con el texto (soporta tildes y ñ)."""
    caja = fuente.getbbox(texto)
    pad = 10
    ancho = (caja[2] - caja[0]) + pad * 2
    alto = (caja[3] - caja[1]) + pad * 2
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 170))
    ImageDraw.Draw(img).text(
        (pad - caja[0], pad - caja[1]), texto, font=fuente, fill=(255, 255, 255, 255)
    )
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def crear_panel(texto, fuente, ancho_max):
    """Panel de texto con salto de línea automático (BGRA)."""
    pad = 12
    medidor = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lineas = []
    for parrafo in texto.split("\n"):
        actual = ""
        for palabra in parrafo.split():
            prueba = (actual + " " + palabra).strip()
            if medidor.textlength(prueba, font=fuente) <= ancho_max - 2 * pad:
                actual = prueba
            else:
                lineas.append(actual)
                actual = palabra
        lineas.append(actual)

    alto_linea = fuente.getbbox("Ayg")[3] + 6
    alto = alto_linea * len(lineas) + pad * 2
    img = Image.new("RGBA", (ancho_max, alto), (0, 0, 0, 170))
    d = ImageDraw.Draw(img)
    for n, linea in enumerate(lineas):
        d.text((pad, pad + n * alto_linea), linea, font=fuente, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def superponer_cartel(frame, img_bgra, cx, cy, opacidad=1.0):
    """Pega una imagen BGRA centrada en (cx, cy), respetando la transparencia."""
    h, w = img_bgra.shape[:2]
    x1, y1 = cx - w // 2, cy - h // 2
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x1 + w, frame.shape[1]), min(y1 + h, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    recorte = img_bgra[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    alfa = recorte[:, :, 3:4].astype(np.float32) / 255.0 * opacidad
    base = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (
        base * (1 - alfa) + recorte[:, :, :3].astype(np.float32) * alfa
    ).astype(np.uint8)


# =============================================================================
# CARGA DE FOTOS (detecta los puntos de cada cara y arma la malla de triángulos)
# =============================================================================
def puntos(landmarks, ancho, alto):
    return np.array([[p.x * ancho, p.y * alto] for p in landmarks[:468]], np.float32)


def preparar_foto(ruta, detector_img):
    # fromfile + imdecode: funciona con tildes y ñ en el nombre del archivo
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        return None

    alto, ancho = img.shape[:2]
    escala = LADO_MAX_FOTO / max(alto, ancho)
    if escala < 1:
        img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    alto, ancho = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = detector_img.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not resultado.face_landmarks:
        return None

    pts = puntos(resultado.face_landmarks[0], ancho, alto)[IDX_MALLA]

    # Margen alrededor de la foto, por si el óvalo de la cara toca el borde
    img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    pts = pts + np.float32(PAD)
    alto_p, ancho_p = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 1, ancho_p - 2)
    pts[:, 1] = np.clip(pts[:, 1], 1, alto_p - 2)

    # Triangulación de Delaunay sobre los puntos de la foto
    subdiv = cv2.Subdiv2D((0, 0, ancho_p, alto_p))
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))
    indice = {(float(p[0]), float(p[1])): i for i, p in enumerate(pts)}

    triangulos = []
    for t in subdiv.getTriangleList():
        vertices = [(float(t[0]), float(t[1])),
                    (float(t[2]), float(t[3])),
                    (float(t[4]), float(t[5]))]
        if all(v in indice for v in vertices):
            triangulos.append([indice[v] for v in vertices])

    return {"img": img, "pts": pts, "triangulos": triangulos}


def cargar_fotos(detector_img):
    extensiones = {".jpg", ".jpeg", ".png", ".webp"}
    fuente = cargar_fuente(TAM_TEXTO)
    fuente_info = cargar_fuente(TAM_INFO)
    fotos = []
    if not CARPETA_FOTOS.exists():
        return fotos
    for ruta in sorted(CARPETA_FOTOS.glob("*")):
        if ruta.suffix.lower() not in extensiones:
            continue
        foto = preparar_foto(ruta, detector_img)
        if foto is None:
            print(f"SIN CARA DETECTADA (se omite): {ruta.name}")
            continue
        texto = ruta.stem.replace("_", " ")
        foto["cartel"] = crear_cartel(texto, fuente)

        info = buscar_texto(ruta.stem)
        if info:
            foto["panel"] = crear_panel(info, fuente_info, ANCHO_INFO)
        else:
            foto["panel"] = None
            print(f"  (sin texto informativo para: {ruta.name})")

        fotos.append(foto)
        print(f"Cargada: {texto}")
    return fotos


# =============================================================================
# DEFORMACIÓN: lleva cada triángulo de la foto sobre el triángulo de tu cara
# =============================================================================
def deformar_cara(foto, pts_dst, forma):
    alto, ancho = forma
    lienzo = np.zeros((alto, ancho, 3), np.uint8)
    img, pts_src = foto["img"], foto["pts"]

    for a, b, c in foto["triangulos"]:
        t1 = pts_src[[a, b, c]]
        t2 = pts_dst[[a, b, c]]

        x1, y1, w1, h1 = cv2.boundingRect(t1)
        x2, y2, w2, h2 = cv2.boundingRect(t2)
        if w1 < 1 or h1 < 1 or w2 < 1 or h2 < 1:
            continue

        # Parte del triángulo que cae dentro de la imagen de la cámara
        cx1, cy1 = max(x2, 0), max(y2, 0)
        cx2, cy2 = min(x2 + w2, ancho), min(y2 + h2, alto)
        if cx1 >= cx2 or cy1 >= cy2:
            continue

        recorte = img[y1:y1 + h1, x1:x1 + w1]
        if recorte.shape[0] != h1 or recorte.shape[1] != w1:
            continue

        try:
            M = cv2.getAffineTransform(
                t1 - np.float32([x1, y1]), t2 - np.float32([x2, y2])
            )
        except cv2.error:
            continue

        deformado = cv2.warpAffine(
            recorte, M, (w2, h2),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        mascara = np.zeros((h2, w2), np.uint8)
        cv2.fillConvexPoly(mascara, np.int32(t2 - np.float32([x2, y2])), 255)

        sub = deformado[cy1 - y2:cy2 - y2, cx1 - x2:cx2 - x2]
        m = mascara[cy1 - y2:cy2 - y2, cx1 - x2:cx2 - x2] > 0
        lienzo[cy1:cy2, cx1:cx2][m] = sub[m]

    return lienzo


# =============================================================================
# FUSIÓN: recorta el óvalo, difumina el borde, iguala la luz y mezcla
# =============================================================================
def a_blanco_y_negro(img, contraste=1.0):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if contraste != 1.0:
        gris = cv2.convertScaleAbs(gris, alpha=contraste, beta=(1 - contraste) * 128)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def fusionar(frame, lienzo, pts_full):
    alto, ancho = frame.shape[:2]
    oval = pts_full[OVALO]

    x, y, w, h = cv2.boundingRect(np.int32(oval))
    margen = int(w * 0.15)
    x0, y0 = max(x - margen, 0), max(y - margen, 0)
    x1, y1 = min(x + w + margen, ancho), min(y + h + margen, alto)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None

    # Máscara del óvalo de la cara, con borde suave
    mascara = np.zeros((y1 - y0, x1 - x0), np.uint8)
    contorno = cv2.convexHull(np.int32(oval - np.float32([x0, y0])))
    cv2.fillConvexPoly(mascara, contorno, 255)

    k = max(3, int(w * SUAVIZADO_BORDE))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mascara = cv2.erode(mascara, kernel)
    mascara = cv2.GaussianBlur(mascara, (k, k), 0)

    roi_f = frame[y0:y1, x0:x1]
    roi_p = lienzo[y0:y1, x0:x1]

    # Igualar solo la luminosidad (el tono de piel de la foto no se toca)
    if IGUALAR_LUZ > 0:
        dentro = mascara > 128
        if dentro.sum() > 200:
            lab_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_p = cv2.cvtColor(roi_p, cv2.COLOR_BGR2LAB).astype(np.float32)
            luz_f = lab_f[:, :, 0][dentro]
            luz_p = lab_p[:, :, 0][dentro]
            ganancia = np.clip(luz_f.std() / (luz_p.std() + 1e-6), 0.7, 1.4)
            nueva = (lab_p[:, :, 0] - luz_p.mean()) * ganancia + luz_f.mean()
            lab_p[:, :, 0] = np.clip(
                lab_p[:, :, 0] * (1 - IGUALAR_LUZ) + nueva * IGUALAR_LUZ, 0, 255
            )
            roi_p = cv2.cvtColor(lab_p.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Blanco y negro (después de que roi_p ya existe)
    if MODO_COLOR in ("bn_cara", "bn_toda"):
        roi_p = a_blanco_y_negro(roi_p, CONTRASTE_BN)

    alfa = (mascara.astype(np.float32) / 255.0 * OPACIDAD)[:, :, None]
    mezcla = roi_f.astype(np.float32) * (1 - alfa) + roi_p.astype(np.float32) * alfa
    frame[y0:y1, x0:x1] = mezcla.astype(np.uint8)
    return x, y, w, h


def dibujar_panel(frame, panel, caja, seg_desde_cambio):
    """Panel de información al costado de la cara, con fundido de entrada."""
    op = (seg_desde_cambio - INFO_RETARDO) / INFO_FADE
    op = min(max(op, 0.0), 1.0)
    if op <= 0:
        return

    alto, ancho = frame.shape[:2]
    x, y, w, h = caja
    ph, pw = panel.shape[:2]

    # Derecha si entra; si no, izquierda
    if x + w + INFO_SEPARACION + pw <= ancho:
        cx = x + w + INFO_SEPARACION + pw // 2
    else:
        cx = x - INFO_SEPARACION - pw // 2

    # Centrado vertical respecto a la cara, sin salirse de la pantalla
    cy = y + h // 2
    cy = min(max(cy, ph // 2 + 5), alto - ph // 2 - 5)

    superponer_cartel(frame, panel, cx, cy, op)


def dibujar_obra(frame, resultado, fotos, indice_base, seg_desde_cambio):
    alto, ancho = frame.shape[:2]
    for i, landmarks in enumerate(resultado.face_landmarks):
        foto = fotos[(i + indice_base) % len(fotos)]
        pts = puntos(landmarks, ancho, alto)
        lienzo = deformar_cara(foto, pts[IDX_MALLA], (alto, ancho))
        caja = fusionar(frame, lienzo, pts)
        if caja is not None:
            x, y, w, h = caja
            cartel = foto["cartel"]
            superponer_cartel(
                frame, cartel, x + w // 2, y + h + cartel.shape[0] // 2 + 8
            )
            if foto.get("panel") is not None:
                dibujar_panel(frame, foto["panel"], caja, seg_desde_cambio)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not MODELO.exists():
        print(f"No encontré el modelo: {MODELO}")
        print("Descargalo como te indiqué (face_landmarker.task) y ponelo junto a este script.")
        return

    base = python.BaseOptions(model_asset_path=str(MODELO))

    # Detector para las fotos (modo imagen)
    opciones_img = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opciones_img) as detector_img:
        fotos = cargar_fotos(detector_img)

    if not fotos:
        print(f"No hay fotos utilizables. Poné imágenes con una cara de frente en: {CARPETA_FOTOS}")
        return

    # Detector para la cámara (modo video)
    opciones_vid = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_CARAS,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if USAR_DSHOW:
        cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Probá cambiar CAMARA_INDEX.")
        return
    if USAR_DSHOW:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAPTURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAPTURA)

    cartel_fijo = (
        crear_cartel(TEXTO_FIJO, cargar_fuente(TAM_TEXTO + 6)) if TEXTO_FIJO else None
    )

    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    pantalla_completa = False
    salto = 0
    t0 = time.time()
    ultimo_ts = 0
    indice_anterior = None
    t_cambio = time.time()

    print("Controles: [q]/[ESC] salir | [f] pantalla completa | [n] siguiente foto")

    try:
        with vision.FaceLandmarker.create_from_options(opciones_vid) as detector:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("No se pudo leer un frame de la cámara.")
                    break
                frame = cv2.flip(frame, 1)
                if MODO_COLOR == "bn_toda":
                    frame = a_blanco_y_negro(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ultimo_ts = max(ultimo_ts + 1, int((time.time() - t0) * 1000))
                resultado = detector.detect_for_video(mp_image, ultimo_ts)

                indice_base = int((time.time() - t0) // CAMBIAR_CADA_SEG) + salto
                if indice_base != indice_anterior:
                    indice_anterior = indice_base
                    t_cambio = time.time()

                dibujar_obra(frame, resultado, fotos, indice_base,
                             time.time() - t_cambio)

                if cartel_fijo is not None:
                    alto, ancho = frame.shape[:2]
                    superponer_cartel(
                        frame, cartel_fijo, ancho // 2,
                        alto - cartel_fijo.shape[0] // 2 - 20,
                    )

                cv2.imshow(NOMBRE_VENTANA, frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                elif tecla == ord("n"):
                    salto += 1
                elif tecla == ord("f"):
                    pantalla_completa = not pantalla_completa
                    cv2.setWindowProperty(
                        NOMBRE_VENTANA, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL,
                    )
    finally:
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()