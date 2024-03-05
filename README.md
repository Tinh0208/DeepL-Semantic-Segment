# DeepL-Semantic-Segment

## Đề tài: 
Xây dựng phần mềm nhận dạng 4 đối tượng giao thông: Người đi bộ, Xe đạp, Xe gắn máy, Xe hơi 
## Yêu cầu:
1. Thu thập dữ liệu
2. Phân lớp các đối tượng trên
3. Phân đoạn (segmentation) đối tượng trong ảnh (có nhiều đối tượng trong ảnh)
4. Triển khai mô hình

***
Repository chứa dataset ảnh và mô hình học sâu
- Thư mục tmp chứa những tập dữ liệu thô
- Thư mục zoo_dataset chứa dữ liệu nhiều lớp, bao gồm:
  - Thư mục train chứa dữ liệu huấn luyện
  - Thư mục valid chứa dữ liệu kiểm định
  - Thư mục test chứa dữ liệu đánh giá
  - Thư mục labels chứa tất cả mask/label của các tập dữ liệu tương ứng 
- File csv lưu tên lớp và label tương ứng
- Mô hình lưu dưới format .py
- Notebook chạy trên colab, dùng để import dữ liệu, huấn luyện & đánh giá mô hình
***
Cấu trúc dataset:
```
zoo_dataset 
├── labels 
│   ├── train
│   │   ├── image.jpg 
│   ├── valid  
│   └── test 
├── train
│   ├── mask.npy
├── valid 
└── test 
```

Số lượng ảnh: 10.000 không bao gồm mask
Nguồn thu thập: Coco-2017, Open Image v7, Cityscapes, Thủ công
Tập train: 600 ảnh
  - Car: 150 ảnh
    - Coco: 75
    - Open V7: 75
  - Bicycle: 150 ảnh
    - Coco: 150
  - Motorcycle: 150 ảnh
    - Coco: 75
    - Open V7: 75
  - Person / Pedestrian: 150 ảnh
Tập validation: 200 ảnh
  - Car: 50 ảnh
    - Coco: 25
    - Open V7: 25
  - Bicycle: 50 ảnh
    - Coco: 50
  - Motorcycle: 50 ảnh
    - Coco: 25
    - Open V7: 25
  - Person / Pedestrian: 50 ảnh
Tập test: 200 ảnh
  - Car: 50 ảnh
    - Coco: 25
    - Open V7: 25
  - Bicycle: 50 ảnh
    - Coco: 50
  - Motorcycle: 50 ảnh
    - Coco: 25
    - Open V7: 25
  - Person / Pedestrian: 50 ảnh

