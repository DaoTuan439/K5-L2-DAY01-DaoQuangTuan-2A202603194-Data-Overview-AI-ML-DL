# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4, `Device: cuda`)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả 3 đều PASS kiểm tra SHA-256 trong notebook)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

- cả bức ảnh vì task = image_classification,Record không có tọa độ,Mỗi ảnh chỉ có một danh sách xếp hạng top-5

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

- Class list đi kèm checkpoint và taxonomy ImageNet-1K (1000 lớp cố định). Model không tự nghĩ ra lớp mới, nó chỉ chọn trong danh sách đó.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

- vì class_id dành cho máy, class_name dành cho người đọc,Cùng một số ID mang nghĩa khác nhau ở taxonomy khác nhau và Tên lớp có thể trùng hoặc gần nghĩa nhau (cab, taxi, car), nên cần taxonomy_name để biết ID và tên đang thuộc danh sách nào.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

- Chọn lớp theo chủ thể chính. Cần định nghĩa rõ "chính" là gì, ví dụ vật chiếm diện tích lớn nhất hoặc nổi bật nhất.Nói rõ được gán 1 nhãn hay nhiều nhãn.Khi hai lớp ngang nhau (ví dụ bus và cab) thì xử lý theo thứ tự ưu tiên nào, hoặc escalate hỏi Lab Coach.
  Có ví dụ đúng và sai đi kèm.

6. Vì sao không được lấy `score` của model làm ground truth?

- score chỉ là độ tự tin của model, không phải sự thật. Ở đây 0.51 nghĩa là model chỉ khoảng "một nửa chắc chắn", và các lớp còn lại vẫn có điểm đáng kể (minibus khoảng 0.16, police_van khoảng 0.08).Model có thể tự tin nhưng sai: trong ảnh này xe buýt chiếm đa số mà model lại chọn cab.Ground truth phải do người gán theo guideline. Nếu lấy prediction làm nhãn, lỗi của model sẽ bị đưa ngược vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- Góc trên-trái của box là (385, 69): cách mép trái 385 px, cách mép trên 69 px, tức là ngang đỉnh đầu người đầu bếp.
- Góc dưới-phải của box là (499, 349): ngang chân người đó.
- Rộng = 498.92 − 385.33 ≈ 114 px; cao = 348.92 − 69.24 ≈ 280 px.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

- Ngưỡng 0.20: JSON không có, vì file chỉ lưu từ 0.35 trở lên
- Ngưỡng 0.60: đếm trong JSON thì còn 6 (person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61).

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Hạ ngưỡng sẽ giữ lại nhiều box hơn, kèm theo nhiều box sai hơn reviewer phải:Xóa box nhầm (false positive),Sửa lớp sai,Gộp các box trùng nhau trên cùng một vật,Kiểm tra kỹ các vật nhỏ và vật ở mép ảnh.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

- Vẽ box sát mép phần nhìn thấy được của vật, không chừa khoảng trống và không đoán phần bị che. Mỗi vật (instance) có một box riêng. Nếu vật bị cắt ở mép ảnh thì box dừng ở mép ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- Vật bị cắt mép: person 0.61, bbox_xyxy = [0.12, 263.18, 61.55, 310.78]. Box chạm mép trái ảnh (x = 0.12), và trên ảnh chỉ thấy một cánh tay.Cách xử lý:

* Cách vẽ box → viết thành quy tắc guideline: chỉ vẽ phần nhìn thấy, box dừng ở mép ảnh. Lý do: trường hợp này lặp lại ở nhiều ảnh nên áp dụng chung được.
* Có gán lớp person khi chỉ thấy một bàn tay không → escalate hỏi Lab Coach. Lý do: đây là câu hỏi về định nghĩa lớp mà ảnh không đủ thông tin để người gán nhãn tự quyết. Nếu mỗi người tự quyết thì nhãn sẽ không nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

- Box chỉ là một hình chữ nhật. Nó gồm cả tường, tủ và khoảng trống quanh người.
- Polygon có 348 điểm bám theo dáng người:Bắt đầu ở đỉnh đầu , chỗ đầu hẹp hơn hẳn chiều rộng box. Đi xuống chân trái rồi vòng lên khe giữa hai chân, sau đó đi xuống chân phảiPhần khe giữa hai chân không thuộc người nhưng vẫn nằm trong box. Polygon loại được phần đó ra.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

- instance_id dùng để phân biệt từng vật riêng lẻ trong một ảnh. Ví dụ ảnh có 4 bowl thì cả 4 cùng class_id = 45 nhưng có 4 instance_id khác nhau.instance_id không phải class ID: class ID chỉ cho biết vật đó là loại gì.instance_id không phải tracking ID: nó chỉ có nghĩa trong một ảnh, không theo dõi cùng một vật qua nhiều ảnh hoặc nhiều frame video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

- Mask chỉ phủ các pixel nhìn thấy được của vật, bám sát đường viền. Phần bị vật khác che thì không vẽ. Mỗi pixel chỉ thuộc một instance, nên các mask không được chồng lên nhau

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- Mask dining table 0.58 tô cả lên vùng của các bowl đặt trên bàn (bowl 0.74, 0.70, 0.53). Bàn và bát chạm nhau và mask bị chồng lên nhau. Xử lý bằng guideline: dùng quy tắc ở câu 4, "mỗi pixel chỉ thuộc một vật; vật nằm trên (bowl) được ưu tiên, mask bàn chừa lỗ ở chỗ bát". Tình huống này lặp lại ở nhiều ảnh nên có thể viết thành quy tắc chung, không cần escalate.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                         | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                | Annotator làm gì?                                                                                                 | Reviewer xem gì?                                                                                              |
| --------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | 1 nhãn cho cả ảnh: `class_id` + `class_name` + taxonomy (ImageNet-1K) | Ảnh `traffic` có nhiều xe buýt nhất nhưng model chọn `cab` (0.51), `minibus` chỉ 0.16 → chủ thể chính không rõ                                                                    | Chọn 1 lớp theo quy tắc "chủ thể chính" trong guideline; hai lớp ngang nhau thì escalate                          | Lớp có đúng chủ thể chính và đúng taxonomy không; không lấy prediction làm nhãn                               |
| Phát hiện vật thể     | 1 box `xyxy` (pixel) + 1 lớp cho mỗi vật (COCO-80)                    | `person` 0.61 `[0.12, 263.18, 61.55, 310.78]` chỉ là bàn tay ở mép trái; nồi chảo treo không được phát hiện vì COCO-80 không có lớp này; `oven` 0.63 bên phải giống tủ có bồn rửa | Vẽ box sát phần nhìn thấy, mỗi vật một box; vật bị cắt mép thì box dừng ở mép ảnh; ca "chỉ thấy tay" thì escalate | Box có sát vật không, có bỏ sót hoặc trùng box không, lớp có đúng không; kiểm tra kỹ vật nhỏ và vật ở mép ảnh |
| Instance segmentation | 1 polygon/mask + 1 lớp + 1 `instance_id` cho mỗi vật                  | Mask `dining table` 0.58 đè lên các `bowl` (0.74, 0.70, 0.53); `potted plant` 0.63 thực ra là bó lá khô treo; `spoon` 0.49 thực ra là muôi treo tường                             | Vẽ biên sát vật, mỗi pixel chỉ thuộc một instance, vật nằm trên được ưu tiên                                      | Mask có chồng nhau không, biên có lem ra nền không, `instance_id` có duy nhất không, lớp có đúng không        |

Lab hôm nay chỉ thấy bước **prediction**. Các lỗi trên là đầu vào cho **QC/rework**: reviewer đánh dấu lỗi → cập nhật guideline (ví dụ quy tắc "chỉ thấy tay có tính là `person` không") → annotator sửa **ground truth** → dữ liệu đã sửa dùng để **huấn luyện** lại model.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

- Chỉ dùng 3 ảnh COCO công khai mà notebook cung cấp; không đưa ảnh cá nhân, dữ liệu nội bộ, họ tên hay MSSV vào báo cáo hoặc output

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

- Dừng lại ngay và báo Lab Coach / GV qua kênh hỗ trợ của lớp; không tự upload hoặc chia sẻ dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
