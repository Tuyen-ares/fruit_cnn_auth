# Hướng dẫn cài đặt chi tiết như thư viện, môi trường thực nghiệm,..., cài ứng dụng và cách sử dụng

Môi trường thực nghiêmj trên visual studio code
cách cài đặt :
bước 1: vào link này nhấn download công cụ: <https://code.visualstudio.com/>
![alt text](image.png)

Sau khi tải về : hãy chắc chắn máy bạn có sẵn python hoặc không hãy tải
python từ link:<https://www.python.org/downloads/>

Bước 2: cài đặt extension

- Hãy cài đặt extension market để có thể download các extension khác
 do chúng em dùng python nên extension nên cài đặt python để dễ dàng hoạt động và chạy
- Sau khi cài xong các extension cần thiết thì bước kế là import các thư viện bằng lệnh trên thanh terminal(để bật thanh này bạn hãy nhìn lên phía trên cùng bên trái có dấu 3 chấm, thì nhấn vào sẽ xổ ra thanh terminal), do có file requirements.txt lưu trữ các thư viện cần thiết trong đây thì chúng ta nhập lệnh :  " pip install -r requirements.txt " trên thanh terminal
sau đó cac sthuw viện sẽ được cài đặt và bạn thêm lệnh : " pip install opencv-python --force-reinstall " để hoàn tất việc install thư viện vào môi trường "

Bước 3: chạy code và train mô hình
Đầu tiên các bạn nên tải hình ảnh từ trên kaggle về

- link kaggle :<https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition>
- Ở đây có 3 folder chính đó là train,test và validation và vì trong code chúng em đặt lệnh và tên file có thêm số 1 đằng trước nên khi tải về và bỏ vào thư mục thì đặt tên lần lượt các folder : train1,test1 và validation1 để chương trình có thể đọc hiểu tên folder để bắt đầu train mô hình
- Sau khi thêm folder ảnh đầy đủ thì thực hiện bấm nút run trên file cnn_fruit.py để thực hiện huấn luyện mô hình,sau khi huấn luyện sẽ xuất ra các thông số qua các bức ảnh trực quan, sau cùng ta vào file test.py để thực hiện chạy code sau khi mô hình đã được train, file này sẽ gồm 2 loại xác thực đó là gửi file ảnh từ máy lên và xác thực từ camera, khi chọn xác thực bằng camera thì lấy bất kfi hình ảnh trái cây nào vào gần camera và cho bức ảnh rõ nét nhất có thể thì mô hình sẽ nhận diện vaf thực hiện dự đoán với % dự đoán cao hơn.
#   f r u i t _ c n n _ a u t h  
 