import os
import socket
import time
import pickle
import pandas as pd
from collections import deque
import uuid
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from ForzaDataPacket import ForzaDataPacket
from feature_extraction import extract_features_window

# ================= SESSION =================
SESSION_ID = str(uuid.uuid4())
print(f"🆔 Session started: {SESSION_ID}")

# ================= INFLUXDB =================
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "fh4-test")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "fh4-telemetry")


client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
write_api = client.write_api(write_options=SYNCHRONOUS)

print("📡 Connected to InfluxDB")

# ================= CONFIG =================
UDP_PORT = 5000
WINDOW_SECONDS = 2.0
MIN_FRAMES = 1

LABEL_MAP = {
    0: "aggressive",
    1: "assertive",
    2: "defensive",
    3: "passive"
}

# ================= LOAD MODEL =================
with open("driving_style_modelv1.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES = bundle["features"]

print("✅ Model loaded")
print("🧠 Features expected:", FEATURES)

# ================= UDP SOCKET =================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.settimeout(2.0)

print(f"🎧 Listening for FH4 telemetry on UDP {UDP_PORT}...\n")

# ================= BUFFER =================
buffer = deque()
last_print = 0

# ================= MAIN LOOP =================
while True:
    try:
        raw, addr = sock.recvfrom(2048)

        packet = ForzaDataPacket(raw, "fh4")
        frame = packet.to_dict()

        ts = datetime.utcnow()   # ✅ FIXED
        frame["capture_time"] = ts.timestamp()

        buffer.append(frame)

        cutoff = time.time() - WINDOW_SECONDS
        while buffer and buffer[0]["capture_time"] < cutoff:
            buffer.popleft()

        # -------- TELEMETRY WRITE --------
        telemetry_point = (
            Point("car_telemetry")
            .tag("session_id", SESSION_ID)
            .field("speed", float(frame.get("speed", 0.0)))
            .field("rpm", float(frame.get("current_engine_rpm", 0.0)))
            .field("gear", int(frame.get("gear", 0)))
            .time(ts)   # ✅ datetime, not float
        )
        write_api.write(bucket=INFLUX_BUCKET, record=telemetry_point)

        if len(buffer) < MIN_FRAMES:
            continue

        # -------- ML INFERENCE --------
        df = pd.DataFrame(buffer)
        feats = extract_features_window(df)
        X = pd.DataFrame([feats])

        for col in FEATURES:
            if col not in X.columns:
                X[col] = 0.0
        X = X[FEATURES]

        probs = model.predict_proba(X)[0]
        idx = int(probs.argmax())

        # -------- STYLE WRITE --------
        # -------- ML PREDICTION WRITE (FIXED) --------
        style_point = (
            Point("ml_prediction")                # measurement
            .tag("session_id", SESSION_ID)
            .tag("driving_style", LABEL_MAP[idx]) # human-readable tag
            .field("driving_style_id", idx)       # numeric → Grafana can plot
            .field("confidence", float(probs[idx]))
            .time(ts)
        )

        write_api.write(bucket=INFLUX_BUCKET, record=style_point)


        if time.time() - last_print > 0.5:
            print(
                f"🧠 STYLE: {LABEL_MAP[idx].upper():<10} | "
                f"CONF: {probs[idx]:.2f}"
            )
            last_print = time.time()

    except socket.timeout:
        print("⏳ Waiting for telemetry...")
    except KeyboardInterrupt:
        print("\n🛑 Drive session ended")
        print(f"🆔 Session ID: {SESSION_ID}")
        break
    except Exception as e:
        print("❌ Error:", repr(e))
