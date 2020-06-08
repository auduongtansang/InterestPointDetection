# InterestPointDetection
Phát hiện điểm hấp dẫn, đối sánh ảnh bằng bộ mô tả SIFT

## Cài đặt
  - Tải OpenCV 4.3.0 và giải nén vào C:/OpenCV/
  - Build project ở chế độ debug (hoặc release) x64
  - Chép file opencv_world430d.dll hoặc (opencv_world430.dll) vào thư mục chứa file thực thi (.exe) đã được build
  
## Cách sử dụng
  - Mở Command Prompt hoặc Windows PowerShell và nhập lệnh:
  ```bash
  path/to/executable/file/<exeFile>.exe taskName path/to/input/file/<imageFile>.jpg <taskArgument1> <taskArgument2> ...
  ```
**1. Phát hiện điểm góc:**
  - task = harris
  ```bash
  InterestPointDetection.exe harris test.jpg <coefficient: (0.04 - 0.06)> <threshold: (0.00 - 1.00)>
  ```
**2. Phát hiện blob bằng Laplacian of Gaussian:**
  - task = blob
  ```bash
  InterestPointDetection.exe blob test.jpg <sigma> <step>
  ```
**3. Phát hiện blob bằng Difference of Gaussian:**
  - task = dog
  ```bash
  InterestPointDetection.exe dog test.jpg <sigma> <step> <contrastThreshold> <eRatioThreshold>
  ```
**4. Đối sánh hai ảnh bằng bộ mô tả SIFT:**
  - task = sift
  ```bash
  InterestPointDetection.exe sift test1.jpg <sigma1> <step1> <contrastThreshold1> <eRatioThreshold1> test2.jpg <sigma2> <step2> <contrastThreshold2> <eRatioThreshold2> <distanceThreshold>
  ```
## Giấy phép
  - Phần mềm đơn giản phục vụ mục đích học tập
  - Không vì bất kì lý do kinh doanh nào
