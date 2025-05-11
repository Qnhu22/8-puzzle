# GIỚI THIỆU BÀI TOÁN 8-PUZZLE
8-Puzzle là một bài toán cổ điển trong trí tuệ nhân tạo, bao gồm một bảng vuông 3x3 với 8 ô đánh số từ 1 đến 8 và một ô trống (0). Mục tiêu của bài toán là sắp xếp lại các ô sao cho chúng nằm đúng vị trí theo trạng thái mục tiêu đã định, bằng cách di chuyển các ô kề với ô trống vào vị trí của nó.
# 1. MỤC TIÊU
Mục tiêu chính của dự án là xây dựng một phần mềm trực quan, linh hoạt để giải quyết bài toán 8-Puzzle – một trong những bài toán kinh điển trong lĩnh vực trí tuệ nhân tạo và tối ưu hóa.

Dự án không chỉ giúp người dùng hiểu rõ cách các thuật toán tìm kiếm hoạt động mà còn so sánh trực tiếp hiệu suất giữa chúng thông qua số bước giải và thời gian thực thi. Giao diện trực quan cho phép hiển thị các bước giải, trạng thái trung gian và các trạng thái belief (niềm tin) trong môi trường không xác định, giúp người học dễ dàng quan sát và phân tích quá trình giải của từng thuật toán.

Ngoài ra, dự án còn hướng đến mục tiêu giảng dạy và nghiên cứu, hỗ trợ sinh viên, giảng viên hoặc lập trình viên có thể thử nghiệm, mở rộng hoặc tích hợp các phương pháp AI nâng cao vào bài toán cổ điển này.
# 2. NỘI DUNG
## 2.1 Nhóm thuật toán tìm kiếm không có thông tin (Uninformed Search Algorithms)
Các thành phần chính của bài toán tìm kiếm và giải pháp
- Trạng thái ban đầu:
    Một lưới 3x3 với 8 số từ 1 đến 8 và một ô trống (0), là trạng thái khởi đầu của bài toán ([[1 2 3], [0 5 6], [4 7 8]]).
- Trạng thái mục tiêu:
    Lưới 3x3 với thứ tự số từ 1 đến 8 và ô trống ở vị trí cuối cùng ([[1 2 3], [4 5 6], [7 8 0]]).
- Không gian trạng thái:
    Tập hợp tất cả các cách sắp xếp cụ thể vị trí các ô của lưới 3x3.
- Hành động:
    Ô trống di chuyển lên, xuống, trái, phải để hoán đổi với ô liền kề dựa trên một thuật toán để tìm trạng thái đích
- Chi phí:
    Mỗi bước di chuyển có chi phí bằng 1
- Giải pháp:
    Từ trạng thia sban đầu, tìm ra trạng thái mục tiêu từ các thuật toán tìm kiếm không có thông tin như BFS, DFS, UCS, IDS
  Hình ảnh gif từng thuật toán cùng biểu đồ so sánh
 
  
    
