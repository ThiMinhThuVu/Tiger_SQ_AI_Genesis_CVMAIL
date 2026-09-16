# TIGER SQ-AI — tóm tắt submission theo task và version

**Team:** Genesis_CVMAIL  
**Synapse Team ID:** `3602194`  
**Ngày đối chiếu repo/Synapse:** 2026-09-14

Theo contract hiện tại của challenge:

- **Task 1:** phân đoạn merged/coarse, 16 lớp.
- **Task 2:** phân đoạn full/fine, 31 lớp.
- **Task 3:** dự đoán visibility đa nhãn cho 14 lymph-node stations.

Các điểm dưới đây là **validation nội bộ**, không phải leaderboard score. Trạng
thái `RECEIVED` chỉ xác nhận Synapse đã nhận submission, chưa xác nhận đã chấm
xong.

## Tổng quan version

| Version | Phạm vi | Synapse entity | Submission ID | Trạng thái ghi nhận |
|---|---|---|---|---|
| `v0_test` | Task 1, 2, 3 trong một image | [`syn77264178`](https://www.synapse.org/Synapse:syn77264178) | T1 `9779070`; T2 `9779071`; T3 `9779072` | Early infrastructure test |
| `ver1` | Task 1, 2, 3 trong một image | [`syn77300426`](https://www.synapse.org/Synapse:syn77300426) | T1 `9779568`; T2 `9779569`; T3 `9779570` | `RECEIVED` |
| `ver2-task2` | Chỉ Task 2 | [`syn77315428`](https://www.synapse.org/Synapse:syn77315428) | `9779893` | `RECEIVED` |
| `ver2-task3` | Chỉ Task 3 | [`syn77319129`](https://www.synapse.org/Synapse:syn77319129) | `9779894` | `RECEIVED` |
| `ver2-task1-d517` | Chỉ Task 1, data đã sửa D517 | [`syn77425596`](https://www.synapse.org/Synapse:syn77425596) | `9780545` | Submitted; score pending |
| `ver3-task2-cetl` | Chỉ Task 2, P3 CETL | [`syn77437299`](https://www.synapse.org/Synapse:syn77437299) | `9780598` | Submitted; score pending |
| `ver4-task2-p0` | Chỉ Task 2, P0 | [`syn77437399`](https://www.synapse.org/Synapse:syn77437399) | `9780599` | Submitted; score pending |
| `ver5-task2-p2` | Chỉ Task 2, P2 | [`syn77437553`](https://www.synapse.org/Synapse:syn77437553) | `9780600` | Submitted; score pending |
| `ver3-task3-p2` | Chỉ Task 3, P2 | [`syn77437640`](https://www.synapse.org/Synapse:syn77437640) | `9780601` | Submitted; score pending |
| `ver4-task3-p0` | Chỉ Task 3, P0 | [`syn77437982`](https://www.synapse.org/Synapse:syn77437982) | `9780602` | Submitted; score pending |
| `ver5-task3-p2` | Chỉ Task 3, P2 | [`syn77438364`](https://www.synapse.org/Synapse:syn77438364) | `9780603` | Submitted; score pending |

## `v0_test` — kiểm tra hạ tầng sớm

Archive: `tigersqai_Genesis_CVMAIL_v0_test_task123.tar.gz`  
SHA-256: `5292e73a078796939f208dc48eb004d3e4c6e3167d30a4a9ba0f25305e5a0ba2`

| Task | Nội dung |
|---|---|
| Task 1 | Mask2Former Swin-Small 5-fold; xuất merged/coarse mask. |
| Task 2 | Cùng pipeline segmentation 5-fold; xuất full/fine mask. |
| Task 3 | Visibility MLP 14 nhãn từ pooled pixel-decoder features; xuất xác suất sigmoid liên tục. |

Mục tiêu của version này là kiểm tra image `linux/amd64`, chạy offline, không có
tham số dòng lệnh và ghi đúng `/output/task1`, `/output/task2`,
`/output/task3.csv`. Đây không phải bản cuối.

## `ver1` — submission gộp ba task

Archive: `tigersqai_Genesis_CVMAIL_ver1_task123.tar.gz`  
SHA-256: `530cf04b2e28b8a96aec5762a45ace18e3d34055918daeba906fe1e142cf69d2`

Protocol chung: center-stratified 5-fold, mỗi fold 32 ca train và 8 ca
validation.

| Task | Model/configuration | Local score | Ghi chú |
|---|---|---:|---|
| Task 1 | Mask2Former Swin-Small coarse-only độc lập, P0 `512x896` | `0.76271` | Ensemble 5 checkpoint. |
| Task 2 | Improved-v2 Mask2Former Swin-Small fine, P0 `512x896` | `0.78576` | Ensemble 5 checkpoint. |
| Task 3 | 5 visibility heads ghép đúng fold với encoder Task 2 P0 | `0.78161` | Huấn luyện bằng CSV cũ, 308 frame có nhãn. |

Điểm trung bình nội bộ ba task: `0.77669`. Điểm Task 3 của `ver1` không nên
so sánh trực tiếp với Task 3 CSV mới vì coverage nhãn khác nhau.

## `ver2` — submission tách riêng Task 2 và Task 3

`ver2` không thay thế Task 1. Version này đóng gói riêng hai cấu hình tốt hơn
cho Task 2 và Task 3, mỗi archive chỉ được submit vào đúng task tương ứng.

### `ver2-task2`

- Model: Task 2 P2, Improved-v2 Mask2Former Swin-Small, ensemble 5-fold.
- Độ phân giải: `640x1120`.
- Local score: `0.78691` (`+0.00115` so với Task 2 P0 của `ver1`).
- Output duy nhất: `/output/task2/`.
- Archive: `tigersqai_Genesis_CVMAIL_ver2_task2.tar.gz`.
- SHA-256: `5f20b447d861a65b5af943349fe07cf3ed5ae1cdf6a614434e2bd53f2918c8d8`.
- Kiểm tra local: đủ 5 checkpoint, cả 5 fold có `status: completed`; checksum hợp lệ.

### `ver2-task3`

- Model: encoder Task 2 P0 `512x896` ghép fold-by-fold với 5 visibility heads.
- Nhãn: CSV cập nhật, 516 frame có nhãn.
- Local score: `0.83440`.
- Output duy nhất: `/output/task3.csv`.
- Archive: `tigersqai_Genesis_CVMAIL_ver2_task3.tar.gz`.
- SHA-256: `55c2e5bc45b767f50f1a221feb3363b06fc1dd8e2b56b4aee0111516da216d94`.
- Kiểm tra local: đủ 5 visibility checkpoint, validation predictions và báo cáo định lượng; checksum hợp lệ.

## Trạng thái hoàn thiện hiện tại

- Bản mới nhất cho **Task 1**: `ver2-task1-d517`, retrain từ snapshot đã sửa
  517 frame. Checkpoint-validation task score trung bình là `0.75659` và
  center-macro checkpoint score trung bình là `0.75940`; báo cáo OOF đầy đủ
  đang chờ Slurm job `606`.
- Các bản bổ sung cho **Task 2** đã submit theo thứ tự: `ver3-task2-cetl`,
  `ver4-task2-p0`, `ver5-task2-p2` vào evaluation `9619534`.
- Các bản bổ sung cho **Task 3** đã submit theo thứ tự: `ver3-task3-p2`,
  `ver4-task3-p0`, `ver5-task3-p2` vào evaluation `9619535`.
- Contract tests trong repo: `3 passed` vào ngày 2026-09-11.
- OCI manifests và application layers của hai archive `ver2` đã được audit.
- Chưa có bằng chứng local rằng evaluation đã chuyển từ `RECEIVED` sang
  `SCORED`; truy vấn live cần tài khoản có quyền `READ_PRIVATE_SUBMISSION`.
- Chưa chạy được `docker run` end-to-end trên host hiện tại vì không có quyền
  truy cập Docker socket.

## Tài liệu nguồn

- [`ver2/README.md`](ver2/README.md)
- [`ver1/README.md`](ver1/README.md)
- [`EMAIL_EARLY_TEST_v0_test.eml`](EMAIL_EARLY_TEST_v0_test.eml)
- [`../reports/TECHNICAL_REPORT_VER1_32x8.md`](../reports/TECHNICAL_REPORT_VER1_32x8.md)
