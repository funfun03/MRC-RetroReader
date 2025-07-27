# Baseline Comparison - Mục 2

## Tổng quan

Scripts này giúp thực hiện mục 2 trong yêu cầu của giảng viên: **So sánh với baseline**

## Files đã tạo

### Config Files (trong `configs/`)

- `baseline_roberta_vanilla.yaml` - RoBERTa base không fine-tune
- `baseline_sketch_only.yaml` - Chỉ train Sketch module
- `baseline_intensive_only.yaml` - Chỉ train Intensive module

### Scripts

- `baseline_comparison.py` - Script chính để chạy tất cả comparisons
- `train_baselines.py` - Train individual baseline models
- `quick_eval.py` - Quick evaluation cho testing

## Cách sử dụng nhanh

### Option 1: Quick Test (Recommended để bắt đầu)

```bash
# Test nhanh với pre-trained models
python quick_eval.py --mode compare
```

### Option 2: Full Pipeline (Cần thời gian training)

```bash
# 1. Train từng baseline (mất thời gian)
python train_baselines.py --baseline sketch
python train_baselines.py --baseline intensive
python train_baselines.py --baseline full

# 2. Chạy comparison
python baseline_comparison.py
```

### Option 3: Evaluation only (Nếu đã có trained models)

```bash
python baseline_comparison.py
```

## Kết quả mong đợi

Script sẽ tạo ra:

- **Comparison table** với metrics (EM, F1, HasAns, NoAns)
- **Improvement analysis** (RetroReader vs baselines)
- **Detailed report** (Markdown format)

## Ví dụ kết quả

```
Model               | EM    | F1    | HasAns_EM | HasAns_F1 | NoAns_EM | NoAns_F1
--------------------|-------|-------|-----------|-----------|----------|----------
RoBERTa_Vanilla     | 0.650 | 0.720 | 0.550     | 0.680     | 0.750    | 0.760
Sketch_Only         | 0.680 | 0.750 | 0.580     | 0.710     | 0.780    | 0.790
Intensive_Only      | 0.720 | 0.780 | 0.650     | 0.750     | 0.790    | 0.810
RetroReader_Full    | 0.760 | 0.820 | 0.690     | 0.790     | 0.830    | 0.850
```

## Insights cho báo cáo

1. **Module Contributions**:

   - Sketch module: Cải thiện NoAns detection
   - Intensive module: Cải thiện HasAns accuracy
   - Combined: Synergistic effect

2. **Architecture Benefits**:
   - Two-stage approach superiority
   - Specialized modules effectiveness

## Troubleshooting

- **Memory issues**: Sử dụng `quick_eval.py` hoặc giảm batch size
- **Training quá lâu**: Sử dụng `--quick-test` flag
- **Config errors**: Check paths trong scripts

## Next Steps

Sau khi hoàn thành mục 2, tiếp tục với:

- Mục 3: Sơ đồ hệ thống chi tiết
- Mục 4: Error analysis
- Mục 5: Background survey
