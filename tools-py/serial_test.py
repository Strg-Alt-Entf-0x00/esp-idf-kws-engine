import argparse
import os
import sys
import time
import serial

def main():
    parser = argparse.ArgumentParser(description="Automated serial console benchmark runner")
    parser.add_argument("--port", default=os.environ.get("ESP_PORT", "COM4"), help="Serial port (default: ESP_PORT or COM4)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1)

    # Wake prompt
    ser.write(b"\r\n")
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    print("\n>>> SENDING: kws benchmark\n", flush=True)
    ser.write(b"kws benchmark\r\n")

    t_end = time.time() + 10.0
    buf = b""
    while time.time() < t_end:
        n = ser.in_waiting
        if n > 0:
            chunk = ser.read(n)
            buf += chunk
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
            if b"============================================================" in buf and buf.count(b"============================================================") >= 2:
                if b"kws_dev>" in buf:
                    break
        time.sleep(0.05)

    ser.close()
    print("\n\nDone.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
