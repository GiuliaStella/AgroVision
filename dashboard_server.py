
from pathlib import Path
import threading
import time

import cv2
import uvicorn
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    str(BASE_DIR / "runs" / "detect" / "runs" / "detect" / "agrovision_100-2" / "weights" / "best.pt")
)

CAMERA_INDEX = 0

app = FastAPI(title="AGROVISION")

app.mount(
    "/assets",
    StaticFiles(directory=str(BASE_DIR / "assets")),
    name="assets"
)

@app.get("/tablet")
def painel_tablet():
    return FileResponse(str(BASE_DIR / "tablet_demo.html"))

lock = threading.Lock()
controle = threading.Lock()


latest_frame = None
camera_thread = None
parar_evento = threading.Event()
parar_evento.set()

stats = {
    "status": "Parado",
    "planta": 0,
    "erva_daninha": 0,
    "fps": 0.0,
}


def camera_loop():
    global latest_frame

    camera = None

    try:
        with lock:
            stats["status"] = "Carregando modelo"

        model = YOLO(MODEL_PATH)

        if parar_evento.is_set():
            return

        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not camera.isOpened():
            with lock:
                stats["status"] = "Erro na camera"
            print("Nao foi possivel abrir a camera USB.")
            return

        ultimo_tempo = time.perf_counter()

        with lock:
            stats["status"] = "Conectado"

        while not parar_evento.is_set():
            ok, frame = camera.read()

            if not ok:
                time.sleep(0.1)
                continue

            result = model.predict(
                frame,
                conf=0.50,
                imgsz=640,
                verbose=False,
            )[0]

            contagens = {
                "planta": 0,
                "erva_daninha": 0,
            }

            if result.boxes is not None:
                for classe in result.boxes.cls.tolist():
                    nome = str(model.names[int(classe)]).lower()
                    nome = nome.replace(" ", "_").replace("-", "_")

                    if nome in contagens:
                        contagens[nome] += 1

            anotado = result.plot()

            agora = time.perf_counter()
            intervalo = agora - ultimo_tempo
            ultimo_tempo = agora
            fps = 1.0 / intervalo if intervalo > 0 else 0.0

            ok, buffer = cv2.imencode(
                ".jpg",
                anotado,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            if ok:
                with lock:
                    latest_frame = buffer.tobytes()
                    stats.update(contagens)
                    stats["fps"] = round(fps, 1)
                    stats["status"] = "Conectado"

    except Exception as erro:
        print(f"Erro no reconhecimento: {erro}")
        with lock:
            stats["status"] = "Erro"

    finally:
        if camera is not None:
            camera.release()

        with lock:
            latest_frame = None
            stats.update({
                "status": "Parado",
                "planta": 0,
                "erva_daninha": 0,
                "fps": 0.0,
            })

        print("Camera liberada.")


@app.post("/start")
def iniciar_reconhecimento():
    global camera_thread

    with controle:
        if camera_thread is not None and camera_thread.is_alive():
            if not parar_evento.is_set():
                return {"status": "Iniciando ou conectado"}

            raise HTTPException(
                409,
                "A camera ainda esta encerrando. Tente novamente.",
            )

        parar_evento.clear()
        camera_thread = threading.Thread(
            target=camera_loop,
            daemon=True,
        )
        camera_thread.start()

    return {"status": "Iniciando"}


@app.post("/stop")
def parar_reconhecimento():
    parar_evento.set()

    with lock:
        stats["status"] = "Parando"
        stats["planta"] = 0
        stats["erva_daninha"] = 0
        stats["fps"] = 0.0

    return {"status": "Parando"}


@app.get("/health")
def health():
    return {"status": "online", "projeto": "AGROVISION"}


@app.get("/api/stats")
@app.get("/status")
def obter_stats():
    with lock:
        return dict(stats)


@app.get("/snapshot.jpg")
def snapshot():
    with lock:
        frame = latest_frame

    if frame is None:
        raise HTTPException(503, "Aguardando imagem da camera")

    return Response(frame, media_type="image/jpeg")


def gerar_video():
    while True:
        with lock:
            frame = latest_frame

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.04)


@app.get("/video")
def video():
    return StreamingResponse(
        gerar_video(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
