import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from ultralytics import YOLO
import numpy as np
import logging
import queue
import asyncio
from bleak import BleakClient, BleakScanner
import threading
import re
import openai

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# HC-08 的 UUID
HC08_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
HC08_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# HC-08 裝置位址
# hc08_address = "9E97AD80-B02C-477C-875B-AE56EE4DB54F"
hc08_address = "A838409C-77BB-F4E9-997C-A19DBF30BC7E"

# 建立 queue 讓 Thread 傳數據回主執行緒
bluetooth_queue = queue.Queue()

# 初始化 Tkinter 視窗
root = tk.Tk()
root.title("YOLO Webcam Detection")

# 設定畫面大小
canvas_width = 800
canvas_height = 400

# 創建畫布來顯示影片
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# 加載 YOLO 模型 11 you only look once
model = YOLO("model.onnx")

# 設定 YOLO 參數
pred_args = {
    "batch": 16,
    "imgsz": 640,
    "conf": 0.85,
}

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 建立 log 訊息區域
log_frame = tk.Frame(root)
log_frame.pack(fill=tk.BOTH, expand=True)

status_label = tk.Label(log_frame, text="AI訊息", font=("Arial", 12, "bold"))
status_label.pack()

log_text = tk.Text(log_frame, height=10, width=5, wrap=tk.WORD)
log_text.pack(fill=tk.BOTH, expand=True)
# log_text.insert(tk.END, "AI訊息\n")  # 初始訊息

# 狀態區域 (顯示移動 & 靜止時間)
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.BOTH, expand=True)

status_label = tk.Label(status_frame, text="寵物狀態", font=("Arial", 12, "bold"))
status_label.pack()

status_text = tk.Text(status_frame, height=5, wrap=tk.WORD)
status_text.pack(fill=tk.BOTH, expand=True)

# 用於記錄物件的移動狀態
object_status = {}  # key: class_id, value: {"position": (x, y), "moving_time": float, "stationary_time": float, "last_update": float}
temperature = 0
humidity = 0
animal = None

def log_message(message):
    """在 log 區域顯示訊息"""
    log_text.insert(tk.END, f"{message}\n")
    log_text.see(tk.END)  # 自動捲動到底部

def update_status_display():
    """更新狀態區域的顯示"""
    status_text.delete(1.0, tk.END)  # 清空狀態顯示
    if not object_status:
        status_text.insert(tk.END, "目前未偵測到物件")
        return

    for class_id, info in object_status.items():
        class_name = model.names.get(class_id, "Unknown")
        moving_time = info["moving_time"]
        stationary_time = info["stationary_time"]
        status_text.insert(tk.END, f"物件: {class_name} | 累積移動: {moving_time:.1f}s | 累積靜止: {stationary_time:.1f}s\n")

def update_frame():
    success, frame = cap.read()
    if success:
        flipped_frame = cv2.flip(frame, 1)
        results = model(flipped_frame, **pred_args)
        annotated_frame = results[0].plot()

        # 轉換 OpenCV 影像為 PIL 影像
        image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((canvas_width, canvas_height))
        photo = ImageTk.PhotoImage(image)

        # 更新畫布影像
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

        # 更新物件狀態
        current_time = time.time()
        detection_threshold = 10  # 座標變化閾值，超過視為移動

        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                values = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = values
                class_id = int(box.cls[0].item())  # 物件類別 ID
                class_name = model.names.get(class_id, "Unknown")
                global animal
                if animal is None:
                    animal = class_name
                    print(f"animal: {class_name}")

                # 計算物件中心點座標
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if class_id not in object_status:
                    # 初始化該物件的狀態
                    object_status[class_id] = {
                        "position": (cx, cy),
                        "moving_time": 0.0,
                        "stationary_time": 0.0,
                        "last_update": current_time
                    }
                else:
                    prev_cx, prev_cy = object_status[class_id]["position"]
                    distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)  # 計算距離變化
                    elapsed_time = current_time - object_status[class_id]["last_update"]

                    if distance > detection_threshold:
                        # 物件正在移動 -> 累積移動時間
                        object_status[class_id]["moving_time"] += elapsed_time
                    else:
                        # 物件保持靜止 -> 累積靜止時間
                        object_status[class_id]["stationary_time"] += elapsed_time

                    # 更新座標 & 時間戳記
                    object_status[class_id]["position"] = (cx, cy)
                    object_status[class_id]["last_update"] = current_time

        update_status_display()  # 更新狀態區域

    # 設定每 10ms 更新一次畫面
    root.after(10, update_frame)

def capture_image():
    success, frame = cap.read()
    if success:
        flipped_frame = cv2.flip(frame, 1)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        cv2.imwrite(filename, flipped_frame)
        log_message(f"擷取畫面: {filename}")

def image_recognition():
    log_message("=== AI辨識結果 ===")
    if not object_status:
        log_message("目前未偵測到物件")
        return

    for class_id, info in object_status.items():
        class_name = model.names.get(class_id, "Unknown")
        moving_time = info["moving_time"]
        stationary_time = info["stationary_time"]
        # log_message(f"物件: {class_name} | 累積移動: {moving_time:.1f}s | 累積靜止: {stationary_time:.1f}s")

        def api_call():
            global temperature
            global humidity
            if temperature is None or temperature == 0:
                temperature = 25
            if humidity is None or humidity == 0:
                humidity = 25
            openai.api_key = "sk-6mSHBLj3QjA49HtK6756985c692c496dBd2764Ea42B60b41"
            openai.base_url = "https://free.v36.cm/v1/"
            openai.default_headers = {"x-foo": "true"}
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"請告訴我飼養的寵物{class_name}適合在溫度{temperature}, 濕度{humidity}的環境下活動嗎？對於這樣的飼養環境有何建議？" +
                                   f"{class_name}一天的活動時間為{moving_time * 3600}秒，休息時間為{stationary_time * 3600}秒是健康的嗎？",
                    },
                ],
            )
            print(
                f"Ask AI : 請告訴我飼養的寵物{class_name}適合在溫度{temperature}, 濕度{humidity}的環境下活動嗎？對於這樣的飼養環境有何建議？" +
                f"{class_name}一天的活動時間為{moving_time * 3600}秒，休息時間為{stationary_time * 3600}秒是健康的嗎？")
            log_message(completion.choices[0].message.content)

        threading.Thread(target=api_call, daemon=True).start()


def process_bluetooth_data():
    while not bluetooth_queue.empty():
        data = bluetooth_queue.get()
        print(f"📡 收到藍牙數據: {data}")
        temp_match = re.search(r"溫度:\s*([\d.]+)", data)
        humidity_match = re.search(r"濕度:\s*([\d.]+)%", data)
        global temperature
        global humidity
        temperature_val = float(temp_match.group(1)) if temp_match else None
        humidity_val = float(humidity_match.group(1)) if humidity_match else None
        if temp_match:
            temperature = temperature_val
            print(f"解析出的溫度: {temperature}°C")
        if humidity_match:
            humidity = humidity_val
            print(f"解析出的濕度: {humidity}%")
    root.after(100, process_bluetooth_data)


def bluetooth_listener():
    print("🔍 掃描藍牙設備...")
    devices = asyncio.run(BleakScanner.discover())
    print(f"🔍 發現 {len(devices)} 個裝置")

    hc08_address = "9E97AD80-B02C-477C-875B-AE56EE4DB54F"
    for device in devices:
        if "HC-08" in (device.name or ""):
            hc08_address = device.address
            print(f"🎯 找到 HC-08: {hc08_address}")
            break

    print("🔗 嘗試連接 HC-08...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def connect():
        async with BleakClient(hc08_address) as client:
            print(f"✅ 已連線到 {hc08_address}")

            def notification_handler(sender, data):
                message = data.decode("utf-8")
                bluetooth_queue.put(message)  # 把數據放入 queue

            await client.start_notify(HC08_CHARACTERISTIC_UUID, notification_handler)

            while True:
                await asyncio.sleep(1)

    loop.run_until_complete(connect())

# 創建一個水平排列的框架來容納按鈕
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# 創建「擷取畫面」按鈕
# capture_button = ttk.Button(button_frame, text="擷取畫面", command=capture_image)
# capture_button.pack(side=tk.LEFT, padx=10)

# 創建「影像辨識」按鈕
test_button = ttk.Button(button_frame, text="AI分析照顧", command=image_recognition)
test_button.pack(side=tk.LEFT, padx=10)


bt_thread = threading.Thread(target=bluetooth_listener, daemon=True)
bt_thread.start()

# 啟動畫面更新
update_frame()

# 啟動藍芽
process_bluetooth_data()

# 運行 Tkinter 事件迴圈
root.mainloop()

# 釋放攝影機資源
cap.release()
cv2.destroyAllWindows()
