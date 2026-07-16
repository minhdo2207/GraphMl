# 5. Thảo luận

## 5.1 Phân tích kết quả chính

**Vai trò của cấu trúc đồ thị:**
- MLP chỉ đạt 0.5888, trong khi các mô hình GNN đạt 0.81-0.83 → cải thiện **+24%**
- Kết quả này xác nhận tầm quan trọng của thông điệp đồ thị (message passing) cho bài toán phân loại node
- Đặc trưng ngữ nghĩa (citation context 1433-dim) một mình không đủ, cần kết hợp cấu trúc trích dẫn

**So sánh các kiến trúc GNN:**
- GCN (0.8276) và GAT (0.8272) có hiệu suất tương đương, phù hợp với lý thuyết: attention mechanism không cải thiện đáng kể trên Cora
- GraphSAGE (0.8130) thấp hơn ~1.5%, có thể do aggregator mean không tối ưu bằng GCN/GAT trên đồ thị thưa
- Mô hình đề xuất (0.8326) đạt kết quả cao nhất, vượt GAT **+0.5%**

**Độ ổn định:**
- Standard deviation thấp (0.005-0.014) trên 5 seeds → training ổn định, không phụ thuộc nhiều vào random seed
- Val-test gap nhất quán (~1-2%) → không có hiện tượng overfitting nghiêm trọng

## 5.2 Phân tích ablation study

**Tác động của residual skip connection:**
- Thêm residual: 0.8164 → 0.8334 (**+1.7%**)
- Residual cho phép gradient flow trực tiếp từ output về input features, giảm vanishing gradient
- Đặc biệt quan trọng với GNN nông (2 layers) nhưng vẫn có tác dụng rõ rệt

**Tác động của DropEdge:**
- DropEdge (p=0.2): 0.8334 → 0.8326 (**-0.1%**, không đáng kể)
- Trên Cora (đồ thị nhỏ, 2708 nodes), DropEdge không cung cấp regularization hiệu quả
- Có thể do mô hình đã đủ capacity, không bị overfitting → regularization không cần thiết

**Tác động của số attention heads:**
- Giảm heads từ 8 → 4: 0.8326 → 0.8260 (**-0.7%**)
- Nhiều heads cho phép học nhiều subspace attention khác nhau, cải thiện biểu diễn
- 8 heads là lựa chọn phù hợp cho Cora

## 5.3 So sánh với kỳ vọng

**Phù hợp kỳ vọng:**
- ✓ Cấu trúc đồ thị cải thiện đáng kể so với MLP
- ✓ Residual connection có tác dụng tích cực
- ✓ Nhiều attention heads cải thiện hiệu suất

**Không phù hợp kỳ vọng:**
- ✗ DropEdge không có tác dụng rõ rệt (kỳ vọng sẽ cải thiện generalization)
- ✗ Chênh lệch GCN vs GAT không đáng kể (kỳ vọng attention mechanism sẽ vượt trội)

**Giải thích:**
- Cora là dataset nhỏ, đồng nhất → các mô hình GNN đơn giản đã đủ tốt
- Attention mechanism có thể không tạo ra khác biệt lớn khi graph structure đã đủ rõ ràng
- DropEdge hiệu quả hơn trên đồ thị lớn, phức tạp hoặc mô hình sâu hơn

---

# 6. Hạn chế và Hướng phát triển

## 6.1 Hạn chế

**Dataset:**
- Chỉ thử nghiệm trên Cora (1 dataset nhỏ, 2708 nodes, 7 classes)
- Chưa đánh giá trên các dataset lớn hơn (Citeseer, Pubmed) hoặc đồ thị heterogenous
- Kết quả có thể không tổng quát cho các miền dữ liệu khác

**Kiến trúc mô hình:**
- Chỉ thử nghiệm GNN 2 layers, chưa khám phá kiến trúc sâu hơn
- Chưa so sánh với các kiến trúc hiện đại hơn (Graph Transformer, GNN với normalization layers)
- Chưa tối ưu hyperparameters một cách hệ thống (learning rate, weight decay, dropout rate)

**Regularization:**
- DropEdge không có tác dụng, chưa thử nghiệm các phương pháp regularization khác
- Chưa phân tích kỹ现 tượng overfitting trên các subsets khác nhau của data
- Chưa đánh giá uncertainty estimation hoặc calibration

**Đánh giá:**
- Chỉ sử dụng accuracy metric, chưa đánh giá precision, recall, F1-score cho từng class
- Chưa phân tích confusion matrix để hiểu mô hình nhầm lẫn giữa các class nào
- Chưa thử nghiệm trên bài toán thực tế với data noise hoặc missing features

## 6.2 Hướng phát triển

**Mở rộng dataset:**
- Thử nghiệm trên Citeseer, Pubmed, Arxiv để kiểm chứng tính tổng quát
- Thử nghiệm trên OGB datasets (đồ thị lớn, realistic)
- Đánh giá trên đồ thị heterogenous (nhiều loại node, nhiều loại edge)

**Cải tiến kiến trúc:**
- Kết hợp với Graph Transformer (attention over global graph)
- Thêm các techniques hiện đại: Jumping Knowledge, Graph Normalization
- Thử nghiệm mô hình sâu hơn (3-5 layers) với residual connections phức tạp hơn

**Regularization nâng cao:**
- Thử nghiệm Mixup, Label Smoothing, Early Stopping
- Kết hợp DropEdge với các phương pháp khác (DropNode, DropFeature)
- Phân tích effect của regularization strength trên các dataset khác nhau

**Ứng dụng thực tế:**
- Áp dụng cho bài toán thực tế: recommendation systems, social network analysis
- Đánh giá robustness với adversarial attacks
- Thử nghiệm với dynamic graphs (đồ thị biến đổi theo thời gian)

**Phân tích sâu hơn:**
- Visualization attention weights để hiểu mô hình học được gì
- Phân tích feature importance: đặc trưng nào quan trọng nhất
- Evaluation thêm các metrics: AUC-ROC, NDCG cho ranking tasks
