# Baseline Comparison Pipeline for RetroReader

## Hướng dẫn chạy Baseline Comparison (Mục 2)

### Bước 1: Chuẩn bị môi trường

Đảm bảo bạn đã cài đặt đủ dependencies:

```bash
pip install -r requirements.txt
```

### Bước 2: Training các Baseline Models

#### 2.1 Train chỉ Sketch module

```bash
python train_baselines.py --baseline sketch
```

#### 2.2 Train chỉ Intensive module

```bash
python train_baselines.py --baseline intensive
```

#### 2.3 Train full RetroReader (cả 2 modules)

```bash
python train_baselines.py --baseline full
```

**Lưu ý:** Để test nhanh, thêm flag `--quick-test`:

```bash
python train_baselines.py --baseline sketch --quick-test
```

### Bước 3: Chạy Evaluation và So sánh

Sau khi đã train xong các models, chạy script comparison:

```bash
python baseline_comparison.py
```

Hoặc để test nhanh:

```bash
python baseline_comparison.py --quick-test
```

### Bước 4: Phân tích kết quả

Script sẽ tự động tạo ra:

1. **Comparison Table** (`outputs/baseline_comparison/comparison_table_YYYYMMDD_HHMMSS.csv`)
2. **Detailed Results** (`outputs/baseline_comparison/all_results_YYYYMMDD_HHMMSS.json`)
3. **Baseline Report** (`outputs/baseline_comparison/baseline_report_YYYYMMDD_HHMMSS.md`)

### Các Metrics được đo lường:

- **EM (Exact Match)**: Tỷ lệ câu trả lời chính xác hoàn toàn
- **F1 Score**: Điểm F1 trên token level
- **HasAns_EM/F1**: Performance trên câu hỏi có đáp án
- **NoAns_EM/F1**: Performance trên câu hỏi không có đáp án

### Cấu trúc files được tạo:

```
outputs/
└── baseline_comparison/
    ├── comparison_table_YYYYMMDD_HHMMSS.csv
    ├── all_results_YYYYMMDD_HHMMSS.json
    ├── baseline_report_YYYYMMDD_HHMMSS.md
    ├── RoBERTa_Vanilla/
    │   └── metrics.json
    ├── Sketch_Only/
    │   └── metrics.json
    ├── Intensive_Only/
    │   └── metrics.json
    └── RetroReader_Full/
        └── metrics.json
```

### Troubleshooting:

1. **Memory Issues**:

   - Giảm batch size trong config files
   - Sử dụng `--quick-test` flag
   - Set `no_cuda: True` nếu không có GPU

2. **Training quá lâu**:

   - Giảm `num_train_epochs` trong config
   - Sử dụng subset của data bằng cách uncomment các dòng trong scripts

3. **Config file not found**:
   - Đảm bảo các config files đã được tạo trong `configs/` folder
   - Check đường dẫn trong scripts

### Expected Results:

Dự kiến kết quả theo thứ tự performance:

1. **RetroReader Full** (cao nhất) - synergy của cả 2 modules
2. **Intensive Only** - strong QA capabilities
3. **Sketch Only** - good answerability detection
4. **RoBERTa Vanilla** (thấp nhất) - no fine-tuning

### Insights cho báo cáo:

1. **Module Contribution Analysis**:

   - So sánh Full vs Sketch Only → đo contribution của Intensive module
   - So sánh Full vs Intensive Only → đo contribution của Sketch module

2. **Synergistic Effects**:

   - Full model performance vs sum of individual modules
   - Đặc biệt quan trọng cho NoAns questions

3. **Architecture Benefits**:
   - Two-stage approach vs single-stage
   - Specialized modules vs general purpose model

Kết quả này sẽ support cho phần Discussion trong báo cáo về effectiveness của RetroReader architecture.
