# BÁO CÁO CÔNG VIỆC THÁNG 8/2026

**Dự án:** TIGER SQ-AI  
**Thời gian tổng hợp:** 01/08/2026–27/08/2026  
**Phạm vi:** mã nguồn, cấu hình, kết quả thí nghiệm và tài liệu hiện có trong repository

> Các chỉ số trong báo cáo là kết quả local/OOF hoặc held-out test nội bộ. Đây không phải kết quả hidden test hay leaderboard chính thức.

## 1. Tổng quan tiến độ

Trong tháng 8, tôi tập trung vào hai hướng chính: (1) mở rộng và kiểm tra thực nghiệm segmentation trên TIGER SQ-AI/CholecSeg8k; (2) xây dựng SAFE-Graph để phát hiện lỗi segmentation nguy hiểm bằng confidence, thông tin giải phẫu và knowledge graph. Tôi đã hoàn thiện phần lớn hạ tầng thực nghiệm, protocol chống leakage, bộ dữ liệu lỗi OOF, các baseline verifier và nhiều ablation. Kết quả quan trọng nhất là xác định được baseline mạnh, đồng thời loại bỏ sớm các hướng visual/knowledge chưa tạo giá trị tăng thêm một cách ổn định.

## 2. Task, target và tiến độ đạt được

### Task 1 — Tổng hợp và kiểm tra kết quả các mô hình TIGER SQ-AI

**Target**

- Chuẩn hóa cách so sánh các biến thể Mask2Former trên cùng protocol 5-fold theo case.
- Xác định mô hình/cơ chế tốt nhất cho Task 1 và Task 2.
- Phân tích ảnh hưởng của context head, optimal transport và các penalty theo ngữ cảnh.

**Tiến độ đạt được: Hoàn thành giai đoạn phân tích (100%)**

- Đã tổng hợp so sánh C3/C4/C5 và các ablation C4-A0/A1/A2/A3.
- Đã xác định **C4-A1** là cấu hình held-out tốt nhất trong nhóm đã phân tích, đạt `Task 1+2 = 0,74753 ± 0,04051`.
- Đã chỉ ra context auxiliary head có lợi hơn việc can thiệp trực tiếp vào target mass; khác biệt giữa các mô hình chưa đủ mạnh để khẳng định ý nghĩa thống kê do chỉ có 5 folds.
- Đã hoàn thiện báo cáo Task 1 tuned v1: Fine Dice tăng từ `0,707978 ± 0,027124` lên `0,712480 ± 0,027750`, tương ứng `+0,004501`.
- Đã lưu báo cáo cơ chế, lỗi theo class/case và khuyến nghị thí nghiệm tiếp theo.

### Task 2 — Thử nghiệm và audit dữ liệu CholecSeg8k

**Target**

- Xây dựng split theo video để tránh frame leakage.
- Huấn luyện và đánh giá GraSP/Mask2Former và SurgiGraph trên 5 folds.
- Kiểm tra tính hợp lệ của ground truth trước khi dùng kết quả cho công bố.

**Tiến độ đạt được: Hoàn thành vòng diagnostic; thực nghiệm chính thức cần chạy lại (75%)**

- Đã thiết lập dữ liệu gồm **8.080 frames từ 17 videos**, với fixed test 1.600 frames từ 3 videos và development set 6.480 frames.
- Đã hoàn thành đánh giá 5-fold cho GraSP/Mask2Former-R50 và SurgiGraph-Q A3-r2-GF.
- SurgiGraph đạt fixed-test `Task 1 score = 0,7854 ± 0,0070`, `Fine Dice = 0,6804 ± 0,0095`, `Fine nHD = 0,1096 ± 0,0050` trên target diagnostic hiện tại.
- Đã tạo qualitative analysis cho các mẫu best/median/worst.
- Đã phát hiện lỗi quan trọng: pipeline đang đọc `*_endo_mask.png` thay vì dense target `*_endo_watershed_mask.png`. Vì vậy các metric hiện tại chỉ có giá trị diagnostic, chưa được dùng như kết quả semantic segmentation chính thức.
- Đã xác định việc còn lại: sửa bước chuẩn bị dense mask, audit LUT/label, tạo artifact mới không ghi đè dữ liệu cũ và train lại các mô hình trên cùng evaluator.

### Task 3 — Thiết kế hướng nghiên cứu SAFE-Graph

**Target**

- Xây dựng một verifier có thể cảnh báo lỗi segmentation nguy hiểm dựa trên anatomy, uncertainty và quan hệ ngữ cảnh.
- Khóa trước protocol, endpoint, safety gate và nguyên tắc chống leakage.
- Chỉ mở correction/loss khi warning verifier chứng minh được hiệu quả ổn định.

**Tiến độ đạt được: Hoàn thành thiết kế và khóa protocol (100%)**

- Đã viết kế hoạch nghiên cứu SAFE-Graph, xác định research question, giả thuyết có thể bác bỏ, primary endpoint và các phase triển khai.
- Đã quy định dữ liệu test không được dùng trong phát triển verifier; toàn bộ thí nghiệm hiện tại dùng OOF validation.
- Đã khóa nguyên tắc safety: verifier chỉ ở chế độ warning-only; không tự động xóa/sửa mask khi chưa qua clinical review và safety gate.
- Đã xác định station là context mềm, không phải hard rule về sự xuất hiện của anatomy.

### Task 4 — Xây dựng Knowledge Graph và bộ review lâm sàng

**Target**

- Tạo knowledge graph từ protocol, label map và thống kê ground truth.
- Chuẩn hóa node, relation, provenance, reliability và fold-safe statistics.
- Chuẩn bị package để chuyên gia lâm sàng kiểm tra các quan hệ.

**Tiến độ đạt được: Hoàn thành phần kỹ thuật; chờ expert review (85%)**

- Đã xây dựng Knowledge Graph v1 và hoàn thành Slurm build job.
- Đã triển khai seed map station–anatomy, anatomy–anatomy geometry, relation metadata và các mức promotion từ documentation-only đến correction-active.
- Đã tạo expert-review package và clinical pilot package gồm **44 OOF component montages** cùng 5 cặp confusion ưu tiên.
- Đã bổ sung unit/regression tests cho builder, dữ liệu review và các evaluator liên quan.
- Trạng thái hiện tại là `PENDING_EXPERT_REVIEW`; knowledge hiện chưa có thẩm quyền lâm sàng để kích hoạt correction hoặc ontology loss.

### Task 5 — Tạo OOF evidence và error corpus cho SAFE-Graph

**Target**

- Tạo nguồn dự đoán không leakage để huấn luyện/đánh giá verifier.
- Xây taxonomy lỗi ở component level, đặc biệt cho các lỗi có nguy cơ lâm sàng.

**Tiến độ đạt được: Hoàn thành (100%)**

- Đã xuất OOF prediction cho **70 frames thuộc 5 validation cases**, gồm prediction, ground truth, top-3 probabilities, entropy và tool probability.
- Đã xây error corpus gồm **3.837 predicted components**, **1.634 GT components**, **673 dangerous predicted-error candidates** và **170 dangerous GT-error candidates**.
- Đã phân loại các lỗi chính: class swap, fragmentation, boundary/partial error, background hallucination, absent-class hallucination và ambiguous false positive.
- Đã nhận diện giới hạn của corpus: số component dự đoán lớn hơn GT khoảng 2,35 lần, nên cần stratify theo diện tích và cần bác sĩ adjudicate trước khi coi đây là clinical ground truth.

### Task 6 — Xây dựng baseline warning verifier Q0–Q2

**Target**

- Kiểm tra liệu confidence, local features và station context có phát hiện dangerous error tốt hơn baseline ngẫu nhiên hay không.
- Đánh giá theo leave-one-case-out và macro-case AUPRC.

**Tiến độ đạt được: Hoàn thành và có quyết định khoa học (100%)**

- Q0 confidence đạt `macro-case AUPRC = 0,2592`, cao hơn prevalence `0,1754`, và là baseline tốt nhất ở giai đoạn Q0–Q2.
- Q1 local logistic đạt `0,2317`; Q2 oracle station-only đạt `0,2249`; Q2 local + station đạt `0,2477`.
- Đã kết luận station co-visibility đơn giản không đủ mạnh để làm detector độc lập hoặc hard rule.
- Đã quyết định giữ confidence làm baseline và chỉ tiếp tục với quan hệ confusion-specific/fold-safe.

### Task 7 — Đánh giá information gain từ visual feature và knowledge feature

**Target**

- Kiểm tra liệu ROI feature từ DINOv2 và quan hệ AA-018 có bổ sung thông tin ngoài P2-DENSE hay không.
- Chỉ tiếp tục joint model khi từng nhánh vượt baseline theo frozen gate.

**Tiến độ đạt được: Hoàn thành ablation; hai nhánh đều NO-GO (100%)**

- Đã khóa **P2-DENSE raw** làm baseline với `macro-case AUPRC = 0,373769`.
- Visual V4 (P2 + ROI) đạt `0,277386`, giảm `0,096383` và không cải thiện ở case nào (`0/5`) — quyết định **NO-GO**.
- Knowledge K3 (P2 + direction + gated AA-018) đạt `0,278268`, giảm `0,095501`, chỉ cải thiện `1/5` cases — quyết định **NO-GO**.
- K4 ungated đạt `0,399984`, tăng `0,026215` và cải thiện `3/5` cases, nhưng confidence interval cắt 0 và semantic audit chưa đạt; chỉ được giữ như tín hiệu khám phá.
- Đã không chạy joint visual-plus-knowledge model vì cả V4 và K3 đều không qua entry gate, qua đó tránh tối ưu tiếp trên tín hiệu nhiễu.

### Task 8 — Phát triển relational verifier thế hệ tiếp theo

**Target**

- Thay one-sided anatomy rule bằng so sánh hai giả thuyết cho từng confusion pair.
- Đánh giá R-LLR và multi-pair contrastive KG bằng protocol khóa trước.

**Tiến độ đạt được: Hoàn thành protocol và vòng exploratory; chưa qua gate (90%)**

- Đã viết và khóa protocol dual-hypothesis relational likelihood ratio (R-LLR) trước khi xem kết quả.
- Đã thiết kế các ablation R0–R6, negative control, context control và station oracle.
- Đã mở rộng sang protocol multi-pair contrastive KG v2.
- Kết quả R-LLR hiện tại được ghi nhận là **NO-GO**, delta khoảng `−0,0199` so với baseline khóa; chưa mở warning lâm sàng, correction hoặc ontology loss.
- Hướng tiếp theo cần redesign endpoint/evidence source và bổ sung clinical review thay vì tiếp tục tinh chỉnh trên 5 cases hiện tại.

### Task 9 — Củng cố khả năng tái lập và kiểm thử

**Target**

- Mỗi thí nghiệm phải có config, job, artifact, evaluator và test tương ứng.
- Bảo đảm fold safety, provenance và không dùng test để chọn mô hình.

**Tiến độ đạt được: Đã hoàn thiện phần lớn hạ tầng (90%)**

- Đã bổ sung nhiều config và Slurm job cho train, evaluate, export OOF và ablation.
- Đã xây test cho knowledge builder, OOF export, error corpus, P2, visual ROI, pair-specific verifier, R-LLR và warning baselines.
- Đã lưu material passport/decision log cho các protocol và báo cáo chính.
- Việc còn lại là gom các lệnh chạy thành một reproduction checklist thống nhất và chạy lại CholecSeg8k sau khi sửa dense target.

## 3. Kết quả nổi bật trong tháng

1. Hoàn thiện được pipeline SAFE-Graph từ OOF prediction → error corpus → baseline verifier → visual/knowledge ablation → relational verifier.
2. Xác định P2-DENSE là baseline warning mạnh nhất hiện tại với `macro-case AUPRC = 0,373769`.
3. Loại bỏ có kiểm soát các hướng chưa hiệu quả: station hard rule, frozen DINOv2 ROI, gated AA-018 và R-LLR hiện tại.
4. Hoàn thành Knowledge Graph v1 về mặt kỹ thuật và chuẩn bị package cho expert review.
5. Phát hiện sớm lỗi target của CholecSeg8k, ngăn việc diễn giải sai metric diagnostic thành kết quả dense segmentation chính thức.
6. Xác định C4-A1 là cấu hình tốt nhất trong nhóm ablation C3/C4/C5 đã đánh giá.

## 4. Công việc chưa hoàn thành và ưu tiên tiếp theo

| Mức ưu tiên | Công việc tiếp theo | Target hoàn thành |
|---|---|---|
| Cao | Sửa pipeline CholecSeg8k sang `*_endo_watershed_mask.png`, audit LUT và train lại | Có bộ kết quả dense segmentation hợp lệ, cùng split và evaluator |
| Cao | Hoàn tất clinical expert review cho KG và error candidates | Xác nhận relation, confusion pair và clinical relevance |
| Cao | Mở rộng OOF/confirmatory cohort ngoài 5 validation cases | Giảm bất định và kiểm tra khả năng khái quát theo case |
| Trung bình | Redesign relational evidence/endpoint sau kết quả R-LLR NO-GO | Chỉ chạy phương án mới khi có evidence source thực sự mới |
| Trung bình | Chuẩn hóa reproduction checklist và bảng theo dõi artifact | Có thể tái chạy toàn bộ pipeline từ config đến report |
| Sau gate | Warning lâm sàng, correction và ontology loss | Chỉ mở khi verifier vượt frozen safety/performance gate |

## 5. Đánh giá chung

Tiến độ tháng 8 đạt tốt ở phần **hạ tầng nghiên cứu, thiết kế protocol, thực nghiệm có kiểm soát và phân tích kết quả**. Giá trị chính không chỉ nằm ở các kết quả dương mà còn ở việc phát hiện lỗi dữ liệu CholecSeg8k và loại bỏ các giả thuyết không vượt baseline trước khi đầu tư thêm compute. Hai nút thắt còn lại là **clinical expert review** và **quy mô cohort còn nhỏ**; vì vậy chưa nên đưa SAFE-Graph sang auto-correction hoặc đưa các metric CholecSeg8k hiện tại vào kết luận chính thức.

## 6. Tài liệu và artifact đối chiếu

- `reports/BAO_CAO_TIEN_DO_DU_AN_TIGER_SQ_AI.md`
- `reports/C3_C4_C5_MODEL_COMPARISON_AND_MECHANISM_ANALYSIS.md`
- `reports/TASK1_MASK2FORMER_SWIN_SMALL_TUNED_V1_JOB127.md`
- `CHOLECSEG8K_EXPERIMENT_REPORT.md`
- `docs/08.08__SAFE_GRAPH_RESEARCH_PLAN.md`
- `docs/08.08__SAFE_GRAPH_KNOWLEDGE_GRAPH_BUILD_PLAN.md`
- `docs/10.08__SAFE_GRAPH_RESULTS_ANALYSIS_AND_NEXT_STEPS.md`
- `docs/13.08__P2_INFORMATION_GAIN_RESULTS_AND_NEXT.md`
- `docs/13.08__RLLR_FROZEN_PROTOCOL.md`
- `docs/13.08__MULTIPAIR_CONTRASTIVE_KG_V2_PROTOCOL.md`
