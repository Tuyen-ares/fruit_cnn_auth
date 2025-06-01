import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Đây là từ điển cấu hình dùng để xác định các tham số chính như kích thước ảnh,
# kích thước lô, số epoch, tốc độ học và đường dẫn dữ liệu cho quá trình huấn luyện CNN
CONFIG = {
    'img_size': 128, 
    'batch_size': 16, 
    'epochs': 100, 
    'learning_rate': 0.0001, #
    'train_path': 'train1',
    'test_path': 'test1',
    'validation_path': 'validation1'
}

# Đây là thiết lập tăng cường dữ liệu dùng để áp dụng các phép 
# biến đổi (co giãn, xoay, lật, điều chỉnh độ sáng) cho ảnh huấn luyện
# nhằm tăng tính đa dạng và tránh quá khớp khi huấn luyện CNN
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    channel_shift_range=20,
    fill_mode='nearest',
    validation_split=0.2
)

# Đây là thiết lập tăng cường dữ liệu cho tập xác thực và kiểm tra dùng để chỉ co giãn ảnh nhằm giữ nguyên tính toàn vẹn của dữ liệu khi đánh giá CNN
val_test_datagen = ImageDataGenerator(rescale=1./255)

print("Đang tải dữ liệu...")

# Đây là tải dữ liệu huấn luyện dùng để đọc ảnh từ thư mục huấn luyện với cấu hình kích thước ảnh,
# kích thước lô và chế độ phân loại, trộn ngẫu nhiên dữ liệu để huấn luyện CNN
training_set = train_datagen.flow_from_directory(
    CONFIG['train_path'],
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Đây là tải dữ liệu xác thực dùng để đọc ảnh từ thư mục huấn luyện 
# (phân tập xác thực) với cấu hình tương tự, không trộn dữ liệu để đánh giá CNN
validation_set = train_datagen.flow_from_directory(
    CONFIG['train_path'],
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Đây là tải dữ liệu xác thực riêng (nếu có) dùng để đọc ảnh từ thư mục xác thực với 
# cấu hình tương tự, không trộn dữ liệu để đánh giá CNN
if os.path.exists(CONFIG['validation_path']):
    validation_set = val_test_datagen.flow_from_directory(
        CONFIG['validation_path'],
        target_size=(CONFIG['img_size'], CONFIG['img_size']),
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )

# Đây là tải dữ liệu kiểm tra dùng để đọc ảnh từ thư mục kiểm tra 
# với cấu hình tương tự, không trộn dữ liệu để đánh giá hiệu suất cuối cùng của CNN
test_set = val_test_datagen.flow_from_directory(
    CONFIG['test_path'],
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False
)

# Đây là danh sách các lớp dùng để hiển thị các lớp được phát hiện từ 
# tập huấn luyện, phục vụ cho phân loại đa lớp của CNN
classes = list(training_set.class_indices.keys())
print(f"Các lớp được phát hiện: {classes}")

# Đây là tính toán trọng số lớp dùng để xử lý dữ liệu không cân bằng bằng cách gán trọng số cao hơn 
# cho các lớp thiểu số, cải thiện hiệu suất CNN trên các lớp này
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(training_set.labels),
    y=training_set.labels
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Trọng số lớp: {class_weight_dict}")

# Đây là lưu chỉ số lớp dùng để lưu ánh xạ tên lớp sang chỉ số vào tệp JSON để sử dụng sau này trong dự đoán của CNN
with open('class_indices.json', 'w', encoding='utf-8') as f:
    json.dump(training_set.class_indices, f, ensure_ascii=False, indent=2)

# Đây là thiết lập học chuyển giao với MobileNetV2 
# dùng để khởi tạo mô hình CNN được huấn luyện trước trên ImageNet, 
# loại bỏ lớp đầu ra để tùy chỉnh cho bài toán phân loại
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(CONFIG['img_size'], CONFIG['img_size'], 3),
    include_top=False,
    weights='imagenet'
)

# Đây là tinh chỉnh mô hình dùng để mở khóa các lớp cuối của MobileNetV2 để huấn luyện lại, 
# đóng băng 100 lớp đầu để giữ các đặc trưng cấp thấp, tối ưu hóa CNN cho tập dữ liệu cụ thể
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Đây là kiến trúc mô hình tuần tự dùng để xây dựng mạng CNN với MobileNetV2,
# lớp tổng hợp, các lớp dày đặc, chuẩn hóa lô và dropout để thực hiện phân loại đa lớp
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Đây là biên dịch mô hình dùng để cấu hình CNN với bộ tối ưu Adam, hàm mất mát và các chỉ số đánh giá (độ chính xác, AUC, độ chính xác, độ thu hồi) để huấn luyện và đánh giá
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()

# Đây là thiết lập callback dùng để dừng sớm nếu mất mát xác thực không cải thiện,
# lưu mô hình CNN tốt nhất và giảm tốc độ học nếu mất mát dừng giảm
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_fruit_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

print("Bắt đầu huấn luyện...")

# Đây là huấn luyện mô hình dùng để đào tạo CNN trên tập huấn luyện với xác thực,
# sử dụng callback và trọng số lớp để tối ưu hóa hiệu suất phân loại
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=CONFIG['epochs'],
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\nĐánh giá mô hình trên tập kiểm tra...")

# Đây là đánh giá mô hình dùng để tính toán các chỉ số hiệu suất của CNN 
# trên tập kiểm tra (mất mát, độ chính xác kiểm tra, AUC, độ chính xác nằm trong top, độ thu hồi)
test_loss, test_accuracy, test_top_k, test_auc, test_precision, test_recall = model.evaluate(test_set, verbose=1)
print(f"Mất mát kiểm tra: {test_loss:.4f}")
print(f"Độ chính xác kiểm tra: {test_accuracy:.4f}")
print(f"Độ chính xác Top-K: {test_top_k:.4f}")
print(f"AUC: {test_auc:.4f}, Độ chính xác: {test_precision:.4f}, Độ thu hồi: {test_recall:.4f}")

# Đây là dự đoán và tạo ma trận nhầm lẫn dùng để dự đoán trên tập kiểm tra bằng CNN và 
# hiển thị ma trận nhầm lẫn dưới dạng biểu đồ nhiệt để phân tích hiệu suất phân loại
test_set.reset()
predictions = model.predict(test_set, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes

print("\nBáo cáo phân loại:")
print(classification_report(true_classes, predicted_classes, target_names=classes))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Đây là hàm vẽ lịch sử huấn luyện dùng để vẽ biểu đồ độ chính xác và mất mát của CNN trên tập huấn luyện và xác thực, lưu dưới dạng tệp PNG
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# Đây là lưu mô hình và cấu hình dùng để lưu mô hình CNN đã huấn luyện và từ điển cấu hình vào tệp để sử dụng sau này
model.save("final_fruit_model.keras")
with open('model_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("\nHoàn thành huấn luyện và lưu mô hình!")