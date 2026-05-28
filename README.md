# Credit Risk Prediction & Management - Dự báo và Quản trị Rủi ro Tín dụng

## Giới thiệu

Dự án xây dựng pipeline hoàn chỉnh để dự báo rủi ro vỡ nợ trong cho vay thế chấp, so sánh hiệu quả giữa mô hình Thẻ điểm truyền thống và các thuật toán học máy hiện đại. Dữ liệu sử dụng là bộ HMEQ (Home Equity) với 5.960 hồ sơ vay, tỷ lệ vỡ nợ 19,95%.

Mục tiêu không chỉ dừng lại ở việc tối ưu chỉ số mô hình, mà còn cung cấp nền tảng định lượng cho ba quyết định kinh doanh cốt lõi: **Ai nên được cho vay? Cho vay ở mức nào? Cho vay bằng quy trình gì?**

---

## Pipeline

### 1. Tiền xử lý & Phân tích dữ liệu thiếu (MNAR)

Thay vì impute đơn thuần, dự án phân tích cấu trúc thiếu dữ liệu và phát hiện 8/11 biến có tín hiệu Missing Not At Random có ý nghĩa thống kê. Các missing flag được tạo ra như biến độc lập, khai thác tín hiệu hành vi ẩn trong dữ liệu thiếu.

| Biến thiếu | Số hồ sơ thiếu | Bad rate khi thiếu | Bad rate khi có | Chênh lệch | Ý nghĩa |
|---|---|---|---|---|---|
| VALUE_MISSING | 112 (1,9%) | 93,75% | 18,54% | +75,21 điểm | Vay thế chấp không có định giá - hồ sơ bất thường nghiêm trọng |
| DEBTINC_MISSING | 1.267 (21,3%) | 62,04% | 8,59% | +53,45 điểm | Không khai báo thu nhập - đang che giấu gánh nặng nợ |
| CLAGE_MISSING | 308 (5,2%) | 25,32% | 19,66% | +5,67 điểm | Có thể là khách hàng mới, chưa có lịch sử ổn định |
| DELINQ_MISSING | 580 (9,7%) | 12,41% | 20,76% | -8,35 điểm | Khách hàng mới chưa có lịch sử vi phạm - tín hiệu tốt |
| DEROG_MISSING | 708 (11,9%) | 12,29% | 20,98% | -8,69 điểm | Chưa từng bị báo cáo xấu - tín hiệu tốt |
| YOJ_MISSING | 515 (8,6%) | 12,62% | 20,64% | -8,02 điểm | Người mới đi làm hoặc sinh viên - rủi ro thấp hơn trung bình |
| JOB_MISSING | 279 (4,7%) | 8,24% | 20,52% | -12,28 điểm | Khách hàng tài sản cao từ chối khai báo - rủi ro thấp hơn |
| NINQ_MISSING | 510 (8,6%) | 14,71% | 20,44% | -5,73 điểm | Ít hoạt động tín dụng - có thể là khách hàng thận trọng |

### 2. Feature Engineering

Xây dựng các biến mới dựa trên kiến thức nghiệp vụ tín dụng:

| Biến mới | Công thức | IV | Mức độ | Ý nghĩa |
|---|---|---|---|---|
| DEROG_DELINQ_SUM | DEROG + DELINQ | 0,4901 | Strong | Bắt hiệu ứng cộng dồn vi phạm - vượt cả DELINQ đơn (0,4382) |
| DEROG_DELINQ_MAX | max(DEROG, DELINQ) | 0,4092 | Strong | Bắt trường hợp một loại vi phạm cực độ |
| LOAN_LOG | log(1 + LOAN) | 0,1891 | Medium | Gấp đôi IV so với biến gốc (0,091) - giảm right-skew hiệu quả |
| YOJ_BIN | Phân nhóm YOJ thành 5 khoảng | 0,0809 | Weak | Bắt hiệu ứng ngưỡng thâm niên mà biến liên tục không thể hiện |
| MORTDUE_LOG | log(1 + MORTDUE) | 0,0439 | Weak | Giảm độ lệch phân phối MORTDUE |
| CREDIT_AGE_INQUIRY | CLAGE × NINQ | 0,0370 | Weak | Phát hiện profile lịch sử dài nhưng nhiều truy vấn gần đây |
| LOAN_SQ | LOAN² | 0,0119 | Useless | Không thêm giá trị - bị loại ở tất cả các mô hình |
| MORTDUE_SQ | MORTDUE² | 0,0032 | Useless | Polynomial naive không phù hợp dữ liệu tín dụng |

### 3. WOE Encoding & Information Value

Toàn bộ biến được mã hóa WOE trên tập train và áp dụng nguyên bản lên tập test. Bộ lọc IV loại các biến Useless (IV < 0,02), giữ lại 19 biến cho pipeline mô hình.

### 4. Huấn luyện mô hình

6 mô hình được xây dựng và đánh giá trên cùng bộ dữ liệu (stratified split 80/20, seed=42):

- **Scorecard Full & IV-Filtered** - Logistic Regression trên WOE features, tuân thủ yêu cầu giải thích pháp lý
- **XGBoost** - Backward Feature Selection (30 → 21 biến) + Optuna 20 trials
- **LightGBM** - Backward Feature Selection (30 → 18 biến) + Optuna 20 trials
- **CatBoost** - Backward Feature Selection (30 → 23 biến) + Optuna 20 trials
- **Stacking Ensemble** - Meta-learner XGBoost kết hợp dự báo từ Scorecard, XGBoost, LightGBM

### 5. Hiệu chỉnh xác suất & Phân nhóm rủi ro

Xác suất thô từ các mô hình boosting được hiệu chỉnh bằng Isotonic Regression, sau đó phân thành 3 nhóm rủi ro:

- **LOW** (P < 0,20): Phê duyệt tự động
- **MID** (0,20 - 0,50): Thẩm định thủ công
- **HIGH** (P > 0,50): Từ chối tự động

### 6. Giải thích mô hình (SHAP)

Phân tích SHAP Waterfall ở cấp độ cá thể và SHAP Dependence Plot cho 3 biến quan trọng nhất (DEBTINC, CLAGE, DEROG_DELINQ_SUM), rút ra quy tắc kinh doanh có thể triển khai trực tiếp.

---

## Kết quả

| Mô hình | AUROC | KS | PSI | Recall Bad |
|---|---|---|---|---|
| Scorecard Full | 0,8905 | 0,6517 | 0,0106 | 0,57 |
| XGBoost | 0,9394 | 0,7629 | 0,0190 | 0,69 |
| LightGBM | 0,9450 | 0,7829 | 0,0455 | 0,75 |
| **CatBoost** | **0,9474** | **0,7902** | **0,0386** | **0,78** |
| Stacking Ensemble | 0,9451 | 0,7891 | 0,0609 | 0,72 |

**Khuyến nghị triển khai:**
- CatBoost làm mô hình chấm điểm chính (AUROC cao nhất, Recall Bad tốt nhất)
- Scorecard duy trì song song để tuân thủ Basel/IFRS 9 và giải thích pháp lý
- XGBoost làm challenger model giám sát độ trôi phân phối hàng tháng
- LightGBM phục vụ tính xác suất vỡ nợ theo IFRS 9 (hiệu chỉnh xác suất tốt nhất)

---

## Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Thư viện:** pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, matplotlib, seaborn
- **Môi trường:** Jupyter Notebook

---
## Tác giả
Nguyễn Văn Nam Anh 
---
**Repository chỉ bao gồm dataset, các đồ thị cùng các bảng kết quả chính, và mô tả dự án; code và báo cáo đầy đủ được cung cấp khi có yêu cầu**
