# ultra96_tcp_server.py
# Ultra96 acts as TCP server.
# Receives SENSOR_FRAME from laptop, replies with INFERENCE_RESULT.
# Stays open until Ctrl+C.
# Accepts reconnects. Does NOT disconnect on idle timeouts.

import json
import socket
import ssl
import time
from datetime import datetime
from pathlib import Path

from framing import recv_framed_json, send_framed_json
from ai_wrapper import GestureInferenceEngine

PROTOCOL_VERSION = 1
ULTRA96_ROLE = "ultra96_server"

ULTRA96_SERVER_HOST = "0.0.0.0"
ULTRA96_SERVER_PORT = 5555

# only for loop responsiveness. DO NOT close on timeout.
READ_TIMEOUT_SEC = 1.0

# ---- ANSI colours ----
ENABLE_COLOR = True
C_RESET = "\x1b[0m"
C_GREEN = "\x1b[32m"


def c(s: str, color: str) -> str:
    return f"{color}{s}{C_RESET}" if ENABLE_COLOR else s


def ensure_logs_directory() -> Path:
    logs_directory = Path("logs")
    logs_directory.mkdir(exist_ok=True)
    return logs_directory


def append_message_jsonl(log_path: Path, direction: str, message: dict) -> None:
    record = {
        "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
        "dir": direction,
        "msg": message,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def short_summary(message: dict) -> str:
    t = message.get("type", "?")
    src = message.get("src", "?")
    seq = message.get("seq", "?")
    payload = message.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    extras = []
    for k in (
        "status",
        "gesture_hint",
        "gesture",
        "confidence",
        "samples_seen",
        "samples_needed",
        "pred_id",
        "echo_request_seq",
        "t_ms",
    ):
        if k in payload:
            extras.append(f"{k}={payload[k]}")
    return f"type={t} src={src} seq={seq}" + (" " + " ".join(extras) if extras else "")


def handle_client(client_socket: socket.socket, log_path: Path, engine: GestureInferenceEngine) -> None:
    client_socket.settimeout(READ_TIMEOUT_SEC)
    response_seq = 1

    # ---- Handshake ----
    try:
        hello_msg = recv_framed_json(client_socket)
    except ValueError as e:
        print(f"[ultra96] Bad framed message during HELLO: {e}")
        return
    except Exception as e:
        print(f"[ultra96] Failed to read HELLO: {e}")
        return

    append_message_jsonl(log_path, "RX", hello_msg)
    print(f"[ultra96][RX] {short_summary(hello_msg)}")

    welcome_msg = {
        "type": "WELCOME",
        "src": "ultra96",
        "seq": response_seq,
        "payload": {
            "ok": True,
            "role": ULTRA96_ROLE,
            "proto_ver": PROTOCOL_VERSION,
            "server_time_unix": time.time(),
        },
    }
    response_seq += 1
    send_framed_json(client_socket, welcome_msg)
    append_message_jsonl(log_path, "TX", welcome_msg)
    print(c(f"[ultra96][TX] {short_summary(welcome_msg)}", C_GREEN))

    # Reset sliding window for each new client session
    engine.reset()

    # ---- Loop ----
    while True:
        try:
            request_msg = recv_framed_json(client_socket)
        except socket.timeout:
            continue  # keep connection open
        except ValueError as e:
            print(f"[ultra96] Bad framed message: {e}")
            return
        except Exception as e:
            print(f"[ultra96] RX error: {e}")
            return

        append_message_jsonl(log_path, "RX", request_msg)
        print(f"[ultra96][RX] {short_summary(request_msg)}")

        if request_msg.get("type") != "SENSOR_FRAME":
            continue

        payload = request_msg.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        try:
            result = engine.push_payload(payload)
        except Exception as e:
            response_msg = {
                "type": "INFERENCE_RESULT",
                "src": "ultra96",
                "seq": response_seq,
                "payload": {
                    "proto_ver": PROTOCOL_VERSION,
                    "role": ULTRA96_ROLE,
                    "server_time_unix": time.time(),
                    "echo_request_seq": request_msg.get("seq"),
                    "status": "error",
                    "error": str(e),
                    "t_ms": int(payload.get("t_ms", 0)) if "t_ms" in payload else 0,
                },
            }
            response_seq += 1
            send_framed_json(client_socket, response_msg)
            append_message_jsonl(log_path, "TX", response_msg)
            print(c(f"[ultra96][TX] {short_summary(response_msg)}", C_GREEN))
            continue

        response_msg = {
            "type": "INFERENCE_RESULT",
            "src": "ultra96",
            "seq": response_seq,
            "payload": {
                "proto_ver": PROTOCOL_VERSION,
                "role": ULTRA96_ROLE,
                "server_time_unix": time.time(),
                "echo_request_seq": request_msg.get("seq"),
                "status": result["status"],
                "gesture": result["gesture"],
                "confidence": round(float(result["confidence"]), 4),
                "samples_seen": result["samples_seen"],
                "samples_needed": result["samples_needed"],
                "pred_id": result.get("pred_id"),
                "timing": result.get("timing"),
                "t_ms": int(payload.get("t_ms", 0)) if "t_ms" in payload else 0,
            },
        }
        response_seq += 1

        send_framed_json(client_socket, response_msg)
        append_message_jsonl(log_path, "TX", response_msg)
        print(c(f"[ultra96][TX] {short_summary(response_msg)}", C_GREEN))


def main() -> None:
    logs_directory = ensure_logs_directory()
    log_path = logs_directory / "ultra96_server.jsonl"

    certs_dir = Path("certs")
    ca_file = certs_dir / "ca.crt"
    server_cert = certs_dir / "ultra96.crt"
    server_key = certs_dir / "ultra96.key"

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(server_cert), keyfile=str(server_key))

    # Require the laptop to present a client cert signed by our CA (mTLS)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(ca_file))

    # Create AI engine once
    engine = GestureInferenceEngine(
        bitstream_path="cnn.bit",
        gesture_map_path="gesture_map.json",
    )

    # TODO: can place my AI power stuff here

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ULTRA96_SERVER_HOST, ULTRA96_SERVER_PORT))
    server_socket.listen(5)

    print(f"[ultra96] Listening on {ULTRA96_SERVER_HOST}:{ULTRA96_SERVER_PORT}")

    try:
        while True:
            print("[ultra96] Waiting for laptop client…")
            client_socket, client_addr = server_socket.accept()
            try:
                tls_client_socket = context.wrap_socket(client_socket, server_side=True)
            except ssl.SSLError as e:
                print(f"[ultra96] TLS handshake failed: {e}")
                client_socket.close()
                continue

            print(f"[ultra96] Accepted connection from {client_addr}")

            try:
                handle_client(tls_client_socket, log_path, engine)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ultra96] Session ended: {e}")
            finally:
                try:
                    tls_client_socket.close()
                except Exception:
                    pass
                print("[ultra96] Client disconnected. Ready for reconnect.")

    except KeyboardInterrupt:
        print("\n[ultra96] Stopped by user (Ctrl+C)")
    finally:
        try:
            server_socket.close()
        except Exception:
            pass
        print("[ultra96] Server closed")
        print(f"[ultra96] Log: {log_path}")


if __name__ == "__main__":
    main()