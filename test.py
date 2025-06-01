import numpy as np
import cv2
import json
import time
import os
from tensorflow.keras.models import load_model
from collections import deque
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class FruitRecognitionApp:
    def __init__(self):
        self.load_config()
        self.load_model_and_classes()
        self.setup_prediction_smoothing()
        self.setup_ui_elements()
        self.setup_main_menu()
        
    def load_config(self):
        """Tải cấu hình từ file"""
        try:
            with open('model_config.json', 'r') as f:
                self.config = json.load(f)
                self.img_size = self.config.get('img_size', 128)
        except FileNotFoundError:
            print("Không tìm thấy file cấu hình, sử dụng mặc định")
            self.img_size = 128
            
    def load_model_and_classes(self):
        """Tải mô hình và danh sách lớp"""
        model_files = ['best_fruit_model.keras', 'final_fruit_model.keras', 
                      'best_fruit_model.h5']
        
        self.model = None
        for model_file in model_files:
            try:
                self.model = load_model(model_file)
                print(f"Đã tải mô hình: {model_file}")
                break
            except:
                continue
                
        if self.model is None:
            raise Exception("Không thể tải mô hình nào!")
        
        # Tải class indices
        try:
            with open('class_indices.json', 'r', encoding='utf-8') as f:
                class_indices = json.load(f)
                self.classes = {v: k for k, v in class_indices.items()}
        except FileNotFoundError:
            # Fallback to default classes
            self.classes = {0: 'Chuối', 1: 'Dâu tây', 2: 'Dứa', 
                           3: 'Khế', 4: 'Măng cụt', 5: 'Xoài'}
            
        print(f"Các lớp: {list(self.classes.values())}")
        
    def setup_prediction_smoothing(self):
        """Thiết lập làm mượt dự đoán cho camera"""
        self.prediction_buffer = deque(maxlen=15)
        self.confidence_threshold = 0.6
        self.stable_frames = 0
        self.min_stable_frames = 3
        self.current_prediction = None
        self.current_confidence = 0
        self.last_announced = ""
        self.announcement_cooldown = 0
        
    def setup_ui_elements(self):
        """Thiết lập các phần tử UI cho camera"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        
        # Màu sắc đơn giản
        self.colors = {
            'high_confidence': (0, 255, 0),
            'medium_confidence': (0, 165, 255),
            'low_confidence': (0, 0, 255),
            'background': (0, 0, 0),
            'text': (255, 255, 255)
        }
        
    def setup_main_menu(self):
        """Thiết lập menu chính đơn giản"""
        self.root = tk.Tk()
        self.root.title("Nhận Diện Trái Cây")
        self.root.geometry("400x300")
        
        # Tiêu đề đơn giản
        title_label = tk.Label(self.root, text="Nhận Diện Trái Cây", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Camera button
        camera_btn = tk.Button(self.root, 
                              text="Nhận diện qua Camera",
                              font=("Arial", 12),
                              width=25, height=2,
                              command=self.start_camera_mode)
        camera_btn.pack(pady=10)
        
        # Image button  
        image_btn = tk.Button(self.root,
                             text="Nhận diện từ Ảnh",
                             font=("Arial", 12),
                             width=25, height=2,
                             command=self.start_image_mode)
        image_btn.pack(pady=10)
        
        # Exit button
        exit_btn = tk.Button(self.root,
                            text="Thoát",
                            font=("Arial", 12),
                            width=25, height=2,
                            command=self.exit_application)
        exit_btn.pack(pady=10)
        
        # Hướng dẫn đơn giản
        instructions = tk.Label(self.root,
                               text="Nhấn 'q' để thoát khỏi chế độ camera",
                               font=("Arial", 9))
        instructions.pack(pady=10)
        
    def start_camera_mode(self):
        """Bắt đầu chế độ camera"""
        self.root.withdraw()
        try:
            self.run_camera_recognition()
        except Exception as e:
            messagebox.showerror("Lỗi Camera", f"Không thể khởi động camera: {e}")
        finally:
            self.root.deiconify()
            
    def start_image_mode(self):
        """Bắt đầu chế độ nhận diện ảnh"""
        self.root.withdraw()
        try:
            self.run_image_recognition()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động chế độ ảnh: {e}")
        finally:
            self.root.deiconify()
            
    def setup_camera(self):
        """Thiết lập camera"""
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise Exception("Không thể kết nối camera!")
            
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.vid.set(cv2.CAP_PROP_FPS, 30)
        print("Kết nối camera thành công")
        
    def preprocess_frame(self, frame):
        """Tiền xử lý khung hình cho camera"""
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        processed_frame = np.expand_dims(frame_resized, axis=0) / 255.0
        return processed_frame
        
    def predict_with_confidence(self, frame):
        """Dự đoán với độ tin cậy"""
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        predicted_class = self.classes[predicted_class_idx]
        return predicted_class, confidence, predictions
        
    def smooth_predictions(self, prediction, confidence):
        """Làm mượt dự đoán"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence, False
            
        recent_predictions = [p[0] for p in list(self.prediction_buffer)[-8:]]
        most_common = max(set(recent_predictions), key=recent_predictions.count)
        avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
        
        should_announce = False
        if (avg_confidence >= self.confidence_threshold and 
            most_common != self.last_announced and 
            self.announcement_cooldown == 0):
            
            recent_count = recent_predictions[-5:].count(most_common)
            if recent_count >= 3:
                should_announce = True
                self.last_announced = most_common
                self.announcement_cooldown = 30
        
        if self.announcement_cooldown > 0:
            self.announcement_cooldown -= 1
            
        return most_common, avg_confidence, should_announce
        
    def draw_prediction_info(self, frame, prediction, confidence, predictions, is_new_detection=False):
        """Vẽ thông tin dự đoán đơn giản"""
        height, width = frame.shape[:2]
        
        # Xác định màu theo confidence
        if confidence >= 0.8:
            color = self.colors['high_confidence']
        elif confidence >= 0.6:
            color = self.colors['medium_confidence']  
        else:
            color = self.colors['low_confidence']
            
        # Vẽ khung thông tin đơn giản
        cv2.rectangle(frame, (10, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 100), color, 2)
        
        # Hiển thị tên trái cây
        cv2.putText(frame, f"Trai cay: {prediction}", (20, 40), 
                   self.font, 0.7, (255, 255, 255), 2)
        
        # Hiển thị độ tin cậy
        cv2.putText(frame, f"Do tin cay: {confidence:.1%}", (20, 70), 
                   self.font, 0.6, (255, 255, 255), 1)
        
        # Hướng dẫn thoát
        cv2.putText(frame, "Nhan 'q' de thoat", (10, height - 20), 
                   self.font, 0.5, (255, 255, 255), 1)
        
    def run_camera_recognition(self):
        """Chạy nhận diện qua camera"""
        self.setup_camera()
        
        print("Bắt đầu nhận diện trái cây qua camera")
        print("Nhấn 'q' để quay lại menu chính")
        
        # Reset buffer
        self.prediction_buffer.clear()
        self.last_announced = ""
        self.announcement_cooldown = 0
        
        try:
            while True:
                ret, frame = self.vid.read()
                if not ret:
                    print("Không thể đọc frame từ camera")
                    break
                    
                frame = cv2.flip(frame, 1)
                
                prediction, confidence, predictions = self.predict_with_confidence(frame)
                smooth_pred, smooth_conf, is_new_detection = self.smooth_predictions(
                    prediction, confidence)
                
                if is_new_detection:
                    print(f"Phát hiện: {smooth_pred} - Độ tin cậy: {smooth_conf:.1%}")
                
                self.draw_prediction_info(frame, smooth_pred, smooth_conf, 
                                        predictions, is_new_detection)
                
                cv2.imshow('Nhan dien trai cay', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                    
        except KeyboardInterrupt:
            print("Đã dừng bởi người dùng")
        except Exception as e:
            print(f"Lỗi: {e}")
        finally:
            if hasattr(self, 'vid'):
                self.vid.release()
            cv2.destroyAllWindows()
            print("Đã dọn dẹp camera")
            
    def run_image_recognition(self):
        """Chạy nhận diện từ ảnh với giao diện đơn giản"""
        # Tạo cửa sổ mới đơn giản - tăng kích thước cửa sổ
        image_window = tk.Toplevel(self.root)
        image_window.title("Nhận diện từ Ảnh")
        image_window.geometry("900x800")
        
        # Tiêu đề
        title_label = tk.Label(image_window, text="Nhận diện Trái cây từ Ảnh", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Button chọn ảnh
        select_btn = tk.Button(image_window, text="Chọn Ảnh", 
                              font=("Arial", 12),
                              width=15, height=2,
                              command=lambda: self.select_image(image_window))
        select_btn.pack(pady=10)
        
        # Label hiển thị ảnh (tăng kích thước đáng kể)
        self.image_label = tk.Label(image_window, text="Chưa chọn ảnh", 
                                   font=("Arial", 12),
                                   width=380, height=380,
                                   relief='solid', bd=1)
        self.image_label.pack(pady=20)
        
        # Kết quả
        self.result_label = tk.Label(image_window, text="Chưa có kết quả", 
                                    font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)
        
        # Độ tin cậy
        self.confidence_label = tk.Label(image_window, text="", 
                                        font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Top 3 predictions
        tk.Label(image_window, text="Top 3 dự đoán:", 
                font=("Arial", 11, "bold")).pack(pady=(10, 5))
        
        self.predictions_listbox = tk.Listbox(image_window, height=4,
                                             font=("Arial", 10))
        self.predictions_listbox.pack(pady=5, padx=20, fill='x')
        
        # Button quay lại
        back_btn = tk.Button(image_window, text="Quay lại Menu", 
                            font=("Arial", 12),
                            width=15, height=2,
                            command=image_window.destroy)
        back_btn.pack(pady=10)
        
        # Lưu reference
        self.current_image_window = image_window
        self.current_image_path = None
        
    def select_image(self, parent_window):
        """Chọn ảnh từ file"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.gif'),
            ('All files', '*.*')
        ]
        
        image_path = filedialog.askopenfilename(
            parent=parent_window,
            title="Chọn ảnh trái cây",
            filetypes=file_types
        )
        
        if image_path:
            self.current_image_path = image_path
            self.display_image(image_path)
            self.predict_image(image_path)
            
    def display_image(self, image_path):
        """Hiển thị ảnh đã chọn với kích thước lớn hơn"""
        try:
            pil_image = Image.open(image_path)
            # Tăng kích thước hiển thị lên rất lớn
            display_size = (700, 500)
            
            # Resize ảnh giữ tỷ lệ nhưng fill vào kích thước mong muốn
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Nếu ảnh vẫn nhỏ, resize trực tiếp
            if pil_image.size[0] < 400 or pil_image.size[1] < 300:
                pil_image = pil_image.resize((600, 450), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")
            
    def preprocess_image(self, image_path):
        """Tiền xử lý ảnh cho model"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Không thể đọc ảnh")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            processed_image = np.expand_dims(image_resized, axis=0) / 255.0
            
            return processed_image
            
        except Exception as e:
            raise Exception(f"Lỗi tiền xử lý ảnh: {str(e)}")
            
    def predict_image(self, image_path):
        """Dự đoán loại trái cây từ ảnh"""
        try:
            processed_image = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]
            predicted_class = self.classes.get(predicted_class_idx, f"Class_{predicted_class_idx}")
                
            self.update_results(predicted_class, confidence, predictions)
            
        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", f"Không thể dự đoán ảnh: {str(e)}")
            
    def update_results(self, prediction, confidence, all_predictions):
        """Cập nhật kết quả đơn giản"""
        # Kết quả chính
        result_text = f"Trái cây: {prediction}"
        self.result_label.config(text=result_text)
        
        # Độ tin cậy
        confidence_text = f"Độ tin cậy: {confidence:.1%}"
        self.confidence_label.config(text=confidence_text)
        
        # Top 3 predictions
        self.predictions_listbox.delete(0, tk.END)
        top_indices = np.argsort(all_predictions)[-3:][::-1]
        
        for i, idx in enumerate(top_indices):
            class_name = self.classes.get(idx, f"Class_{idx}")
            prob = all_predictions[idx]
            item_text = f"{i+1}. {class_name}: {prob:.1%}"
            self.predictions_listbox.insert(tk.END, item_text)
            
    def exit_application(self):
        """Thoát ứng dụng"""
        result = messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn thoát?")
        if result:
            self.root.quit()
            self.root.destroy()
        
    def run(self):
        """Chạy ứng dụng chính"""
        print("Khởi động ứng dụng nhận diện trái cây")
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Lỗi ứng dụng: {e}")
        finally:
            print("Đã thoát ứng dụng")

def main():
    """Hàm main"""
    try:
        app = FruitRecognitionApp()
        app.run()
    except Exception as e:
        print(f"Lỗi khởi tạo ứng dụng: {e}")
        messagebox.showerror("Lỗi", f"Không thể khởi tạo ứng dụng: {e}")

if __name__ == "__main__":
    main()