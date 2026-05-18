# Đồ án 2 - Ranking trong Học máy

## 1. Thông tin nhóm

| MSSV | Họ và tên |
|---|---|
| 23120245 | Nguyễn Quang Duy |
| 23120258 | Lưu Trọng Hiếu |
| 23120260 | Văn Đình Hiếu |

**Môn học:** Nhập môn Học máy

**Tên đề tài:** Ranking trong Học máy

---

## 2. Chương sách đã chọn

Nhóm chọn **Chương 9 - Ranking** trong sách *Foundations of Machine Learning*.

Chương này nghiên cứu bài toán học xếp hạng trong học máy. Thay vì chỉ dự đoán nhãn hoặc giá trị cho từng mẫu riêng lẻ, bài toán ranking hướng đến việc sắp xếp các đối tượng theo một thứ tự mong muốn. Đây là bài toán xuất hiện trong nhiều ứng dụng như công cụ tìm kiếm, hệ thống gợi ý, truy xuất thông tin, phát hiện gian lận và các hệ thống ra quyết định.

Các nội dung chính của chương sách bao gồm:

- Bài toán ranking và thiết lập score-based ranking.
- Chặn tổng quát hoá dựa trên margin và Rademacher complexity.
- Ranking với Support Vector Machines, còn gọi là SVM-ranking hoặc Ranking SVM.
- Thuật toán RankBoost cho pairwise ranking.
- Bipartite ranking, ROC curve và AUC.
- Preference-based ranking.
- Một số tiêu chí đánh giá ranking như precision, average precision, DCG và NDCG.

---

## 3. Mô tả những gì đã làm

Trong báo cáo, nhóm trình bày lại nội dung chính của chương Ranking. Các phần lý thuyết đã thực hiện gồm:

- Giới thiệu bài toán xếp hạng và động cơ của Learning to Rank.
- Trình bày hàm điểm $h : X \rightarrow \mathbb{R}$ và cách sinh ranking bằng cách sắp xếp điểm số.
- Mô tả quan hệ ưu tiên theo từng cặp và pairwise ranking loss.
- Phân tích margin trong ranking và empirical margin loss.
- Trình bày chặn tổng quát hoá cho score-based ranking.
- Giải thích SVM-ranking như một bài toán SVM trên vector hiệu giữa hai đối tượng.
- Trình bày RankBoost, cách cập nhật trọng số trên các cặp và chặn lỗi thực nghiệm.
- Phân tích bipartite ranking, ROC curve và AUC.
- Trình bày preference-based ranking, thuật toán sort-by-degree và randomized QuickSort.
- Thảo luận các tiêu chí đánh giá ranking như pairwise accuracy, ranking loss, Kendall tau distance, AUC, MAP, MRR và NDCG.

Về thực nghiệm, nhóm cài đặt từ đầu một mô hình **pairwise learning-to-rank** bằng Python và NumPy. Mô hình sử dụng scoring tuyến tính:


$$s_w(x) = w^T x$$


Mô hình được huấn luyện bằng **pairwise hinge loss** trên dữ liệu tổng hợp:

$$loss_ij(w) = max(0, 1 - y_ij * w^T(x_i - x_j))$$

Pipeline thực nghiệm gồm:

1. Sinh dữ liệu ranking tổng hợp từ một hàm tuyến tính có nhiễu.
2. Tạo các cặp ưu tiên từ điểm xếp hạng thật.
3. Huấn luyện mô hình scoring tuyến tính bằng gradient descent.
4. Đánh giá chất lượng ranking bằng các metric phù hợp.
5. Lưu hình ảnh và bảng kết quả phục vụ báo cáo.

Các metric được sử dụng trong thực nghiệm:

- Pairwise accuracy.
- Ranking loss.
- Kendall tau distance.
- Top-10 overlap.

---

## 4. Những điểm mở rộng so với sách

Ngoài việc trình bày lại nội dung trong chương sách, nhóm có bổ sung một số phần mở rộng để báo cáo dễ hiểu hơn và có tính thực hành hơn.

### 4.1. Giải thích trực giác cho lý thuyết

Nhóm bổ sung diễn giải trực quan cho các khái niệm như scoring function, pairwise preference, ranking loss, margin loss, AUC và Kendall tau distance. Các công thức được giải thích theo ý nghĩa trong bài toán xếp hạng thay vì chỉ trình bày dưới dạng toán học.

### 4.2. Bổ sung ví dụ minh hoạ

Báo cáo có thêm các ví dụ nhỏ để minh hoạ:

- Cách tính margin trong ranking.
- Cách SVM-ranking biến cặp đối tượng thành vector hiệu.
- Cách RankBoost phân loại các cặp thành xếp đúng, xếp sai và hoà.
- Cách tính AUC trên mẫu hữu hạn.
- Quan hệ ưu tiên không bắc cầu trong preference-based ranking.

### 4.3. Phân tích độ phức tạp

Nhóm bổ sung phân tích chi phí tính toán cho một số phương pháp:

- Số lượng cặp trong pairwise ranking tăng theo bậc `O(n^2)`.
- Chi phí huấn luyện pairwise ranking tuyến tính trên toàn bộ cặp là `O(n^2 d)` cho mỗi epoch.
- SVM-ranking với kernel có thể tốn bộ nhớ lớn do cần ma trận kernel trên các cặp.
- Sort-by-degree có độ phức tạp `O(n^2)`.
- Randomized QuickSort có độ phức tạp kỳ vọng `O(n log n)`.

### 4.4. Cài đặt thực nghiệm từ đầu

Phần thực nghiệm là điểm mở rộng quan trọng. Nhóm tự cài đặt pairwise learning-to-rank bằng NumPy, không dùng các thư viện học máy cấp cao cho mô hình chính như scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, PyTorch hoặc JAX.

### 4.5. Khảo sát ảnh hưởng của nhiễu và kích thước mẫu

Ngoài thực nghiệm chính, nhóm thực hiện thêm các thí nghiệm để quan sát:

- Khi độ nhiễu tăng, chất lượng ranking giảm.
- Khi kích thước mẫu tăng, mô hình thường học được ranking ổn định hơn.

---

## 5. Hướng dẫn tái tạo kết quả thực nghiệm

Phần code thực nghiệm nằm trong thư mục `code/`.

### 5.1. Cấu trúc thư mục code

```text
code/
├── run_main.py
├── run_experiments.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── pairs.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── metrics.py
│   ├── visualize.py
│   └── utils.py
├── experiments/
│   ├── noise_experiment.py
│   └── sample_size_experiment.py
├── results/
│   ├── figures/
│   └── tables/
└── tests/
    └── test_metrics.py
```

### 5.2. Cài đặt môi trường

Di chuyển vào thư mục `code/`:

```bash
cd code
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 5.3. Chạy thực nghiệm chính

Chạy lệnh sau trong thư mục `code/`:

```bash
python run_main.py
```

Thiết lập mặc định của thực nghiệm chính:

```text
n_samples = 100
n_features = 2
noise_std = 0.1
epochs = 200
learning_rate = 0.05
regularization = 1e-3
seed = 42
```

Sau khi chạy, chương trình sinh các file kết quả:

```text
results/figures/loss_curve.png
results/figures/ranking_comparison.png
results/tables/main_results.csv
```

Ý nghĩa các file:

- `loss_curve.png`: biểu đồ pairwise hinge loss trong quá trình huấn luyện.
- `ranking_comparison.png`: biểu đồ so sánh ranking thật và ranking dự đoán.
- `main_results.csv`: bảng kết quả thực nghiệm chính.

### 5.4. Chạy toàn bộ thực nghiệm

Để chạy toàn bộ thí nghiệm, gồm thực nghiệm chính, khảo sát nhiễu và khảo sát kích thước mẫu, dùng lệnh:

```bash
python run_experiments.py
```

Các file kết quả được sinh thêm:

```text
results/figures/noise_experiment.png
results/figures/sample_size_experiment.png
results/tables/noise_results.csv
results/tables/sample_size_results.csv
```

Ý nghĩa các file:

- `noise_experiment.png`: biểu đồ ảnh hưởng của nhiễu đến chất lượng ranking.
- `sample_size_experiment.png`: biểu đồ ảnh hưởng của kích thước mẫu đến chất lượng ranking.
- `noise_results.csv`: bảng kết quả thí nghiệm với các mức nhiễu khác nhau.
- `sample_size_results.csv`: bảng kết quả thí nghiệm với các kích thước mẫu khác nhau.

### 5.5. Chạy kiểm thử

Nếu muốn kiểm tra một số hàm metric, có thể chạy:

```bash
python -m pytest tests/
```

Nếu môi trường chưa có `pytest`, cài đặt thêm bằng:

```bash
pip install pytest
```

### 5.6. Ghi chú tái tạo

- Nên chạy các lệnh từ thư mục `code/`.
- Các kết quả được lưu trong thư mục `results/`.
- Code sử dụng random seed mặc định là `42` để hỗ trợ tái tạo kết quả.
- Do dữ liệu được sinh tổng hợp có nhiễu, nếu thay đổi seed hoặc các tham số như `noise_std`, `n_samples`, `epochs`, kết quả có thể thay đổi.

---

## 6. Tóm tắt ngắn

Đồ án trình bày chủ đề **Ranking trong Học máy** dựa trên Chương 9 của sách *Foundations of Machine Learning*. Nhóm tập trung vào các khái niệm score-based ranking, preference-based ranking, SVM-ranking, RankBoost, bipartite ranking, ROC và AUC. Bên cạnh phần lý thuyết, nhóm cài đặt từ đầu mô hình pairwise learning-to-rank tuyến tính bằng NumPy và thực hiện các thí nghiệm để minh hoạ ảnh hưởng của nhiễu và kích thước mẫu đến chất lượng ranking.
