# GeoSurg-IC — kết quả pilot và quyết định trước deadline

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent (execution).
- Ngày hoàn tất thí nghiệm: 14/09/2026 UTC; kiểm tra và kết luận: 15/09/2026 UTC.
- Status: COMPLETED, kiểm tra số liệu và metric parity đã qua; chưa lặp lại efficacy ở fold/seed độc lập.
- Nguồn ý tưởng: mô tả GeoSurg-IC do người dùng cung cấp, không coi là paper đã được xác minh.
- Dữ liệu: D517, fold 0, 32 ca train / 8 ca validation; 106 frame validation, 6 center.
- Phạm vi: Task 2 fine segmentation 31 lớp. Đây là validation đã dùng chọn checkpoint gốc, không phải hidden test.
- Tài nguyên: đúng 1 GPU vật lý (GPU 3), chạy tuần tự; peak allocated VRAM khoảng 4.064 MiB (3,97 GiB).

## Kết luận

**Không đưa GeoSurg-IC vào submission dựa trên pilot này.** GeoSurg-IC cải
thiện so với P2 gốc, nhưng không vượt matched augmentation. Boundary-band
Dice cũng không tốt hơn các đối chứng. Vì vậy giả thuyết lợi ích riêng của
geometry-conditioned interaction regularization **chưa được ủng hộ**.

Không mở rộng fold 1 theo gate đã khóa; không sửa hay nộp lại submission.
Checkpoint P2 hiện có được giữ nguyên. Nhánh augmentation-only là một tín
hiệu đáng thử tiếp sau này, chưa đủ bằng chứng để thay ensemble 5-fold.

## So sánh ở độ phân giải gốc

Probability được nội suy về kích thước ảnh gốc trước argmax; GT không resize.
Các cột dưới đây đều aggregate frame → case → center (mỗi center trọng số bằng nhau).

| Nhánh | Fine Dice ↑ | Fine nHD ↓ | Fine score ↑ | GT-boundary-band Dice ↑ | Worst-center score ↑ |
|---|---:|---:|---:|---:|---:|
| P2 D517 gốc | 0.749904 | 0.169357 | 0.790273 | 0.414835 | 0.735935 |
| Matched augmentation | **0.755260** | **0.163920** | **0.795670** | **0.417850** | 0.745409 |
| GeoSurg-IC | 0.754766 | 0.164792 | 0.794987 | 0.417135 | 0.743950 |
| GT-interface IC, matched count | 0.754605 | 0.166103 | 0.794251 | 0.417447 | **0.745701** |
| Shuffled geometry routing IC | 0.754588 | 0.165222 | 0.794683 | 0.417141 | 0.745224 |

Fine score = (Dice + 1 − nHD)/2. Boundary-band Dice là diagnostic trong
dải quanh GT boundary, bán kính 0.3% đường chéo ảnh; chỉ aggregate các lớp
GT có mặt trong dải bằng clinical weight. Đây **không phải** standard Boundary
IoU. Dải GT cố định giống nhau cho mọi nhánh.

- GeoSurg-IC so với P2 gốc: **+0.004714** fine score (+0.4714 điểm phần trăm).
- GeoSurg-IC so với matched augmentation: **−0.000683** fine score và
  **−0.000716** boundary-band Dice; fine score tốt hơn ở 3/8 ca, kém hơn ở 5/8 ca.
- GeoSurg-IC so với GT-interface IC: +0.000736 fine score, nhưng boundary-band Dice thấp hơn.
- GeoSurg-IC so với shuffled routing: +0.000304 fine score; boundary-band Dice gần như bằng nhau và hơi thấp hơn.

Chênh lệch với shuffled routing nhỏ, không chứng minh geometry routing là
nguyên nhân cải thiện. Không chạy significance test hay kết luận về khả năng
generalize từ một fold validation đã dùng chọn checkpoint.

## Pilot đã triển khai

Khởi tạo mọi nhánh từ đúng cùng checkpoint P2 D517 fold 0 (SHA-256
`2e11fbc29f0ab3e4847ca586d14eb5809374fd95f9fd513da34a3a78905ceeb6`).
Freeze Swin encoder; fine-tune 19,900,431 tham số decoder/head. Input
640×1120, 192 optimizer steps, LR 1e-5, IC weight 0.02, IC mỗi 2 bước.
Giữ supervised Mask2Former + coarse/hierarchy/presence objective; không
thêm lớp hoặc thay đổi inference.

Pseudo-depth được tính cục bộ bằng
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
và chỉ dùng để route loss trong train. Không đưa ảnh lên API. Cache 517 ảnh
mất khoảng 2 phút; mỗi fold chỉ đọc training records của mình.

Chọn một cặp lớp kề nhau từ GT, tạo hai appearance perturbations độc lập
trên hai support không giao nhau. Mixed margin:

    delta = m11 - m10 - m01 + m00
    L_IC = sum(w * delta^2)

Không ép semantic features gần hoặc xa nhau theo depth. Bốn view đều nhận
gradient bằng deterministic replay; phần additive của từng phía không bị
phạt. IC probes và depth model không có mặt trong inference.

GT/shuffle đối chứng bảo toàn số pixel routing của geometry trong từng ảnh;
GT là đối chứng vị trí trên boundary với matched count, không phải hoàn toàn
độc lập với depth về lịch exposure. Shuffling thực hiện trong cùng class pair.
Pilot dùng support toàn bộ lớp trong ảnh, chưa tách connected components.
Routing grid 160×280 có thể bỏ lỡ interface rất nhỏ. Đây là prototype có giới
hạn tính toán, không phải bác bỏ mọi biến thể GeoSurg-IC.

## Kiểm tra và artifacts

- 10 tests qua: additive cancellation, mixed-response replay gradient,
  disjoint perturbations, flat-depth behavior, routing budget/shuffle,
  affine depth normalization, case/center aggregation, native metric parity.
- Cả 4 nhánh đủ 192 bước, cùng checkpoint hash, sample order, routing count;
  không có train/validation case overlap; mỗi tiến trình thấy đúng 1 GPU.
- Native evaluation của 5 model dùng đúng cùng 106 frame và thứ tự ảnh.
- Job 668: cả 4 nhánh hoàn tất trong khoảng 34 phút, tính cả khởi tạo/validation.
- Job 669: native evaluation hoàn tất sau đó, khoảng 22 phút. Không còn job đang chạy.

Files:

- `PROTOCOL.md`: thiết kế/gate đã khóa và lệnh chạy.
- `artifacts/pilot_v1/native/fold_0/comparison.json`: số liệu native đầy đủ theo frame/case/center.
- `artifacts/pilot_v1/{aug,geo,gt,shuffle}/fold_0/best.pt`: checkpoint cuối pilot.
- `artifacts/pilot_v1/{arm}/fold_0/validation.json`: validation tại grid input.
- `artifacts/pilot_v1/{arm}/fold_0/{history,provenance,completion}.json`: log và cấu hình từng nhánh.
- `artifacts/pilot_v1/decision.json`: quyết định NO-GO và metric liên quan.
