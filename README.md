# Unity_Voxel_Classifier_Sentis
**3D메쉬를 복셀화하고, 해당 데이터를 TensorFlow로 3D-CNN을 학습시켜 유니티에서 분류하는 도구**

**AI를 활용하였습니다. 3D메쉬 복셀화부터 프로그램안에서 분류까지의 과정을 구축하는데 중점을 두었습니다.**
---
**이 도구는 크게 3단계로 작동합니다.**
1. **데이터 추출(Unity)** : BatchModelProcessor.cs를 통해 학습시킬 모델의 데이터를 추출
2. **AI모델학습(Python)** : train.py로 데이터를 읽어 3D-CNN모델을 학습시킨 후 유니티용 onnx모델 생성
3. **실시간 분류(Unity)** : VoxelClassifier.cs가 실시간으로 분석하고 결과 출력

## 적용 스크린샷

<div align="center">
  <h2>추출한 데이터 시각화한 결과</h2>
  <img src="https://github.com/user-attachments/assets/072972a4-3f99-42ff-b97b-d3c4fd2a902e" width="800">
</div>

<br>

<div align="center">
  <h2>모델의 학습 성능 검증 위해 직접 만든 2D 이미지를 3D복셀로 변환합니다.</h2>
  <img src="https://github.com/user-attachments/assets/02aee1e0-7980-47e4-988f-1f49bece35f9" width="398">
  <img src="https://github.com/user-attachments/assets/631f5a03-33d8-4ff7-9ea6-301f3ba9e7e4" width="398">
</div>

<br>

<div align="center">
  <h2>도끼모양을 생각하고 만든 복셀메쉬 분석결과</h2>
  <img src="https://github.com/user-attachments/assets/20b0e2a0-3d19-4427-959c-ee33003a0526" width="398">
  <img src="https://github.com/user-attachments/assets/36183184-b687-4364-b98e-8bf4085bafec" width="398">
</div>

<br>

<div align="center">
  <h2>창모양을 생각하고 만든 복셀메쉬 분석결과</h2>
  <img src="https://github.com/user-attachments/assets/2b2af559-73fd-450b-b05c-a73e8881bea6" width="398">
  <img src="https://github.com/user-attachments/assets/54e8a28e-e57f-4588-8c49-0f7182538651" width="398">
</div>

## 기능 설명

**BatchModelProcessor.cs(Editor 전용)**

Unity에서 3D 메쉬를 읽어 학습용 데이터로 변환합니다.

Voxelizer스크립트를 호출하여 3D메쉬를 사전 설정된값에 따라 3D 배열로 변환합니다.

변환된 복셀 데이터 중 속이 차 있는 지점의 X,Y,Z좌표만 추출하여 기록합니다.

**train.py**

데이터를 분류하기 위한 3DCNN모델을 학습시킵니다.

지정된 디렉터리에서 txt파일들을 읽어옵니다.

데이터 불균형 및 부족 문제를 해결하기 위해 데이터 증강하기 위해 회전, 반전, 노이즈등을 사용했습니다.

학습 후 ONXX파일 포맷으로 모델을 내보냅니다.

**Voxelizer.cs**

3D메쉬 데이터를 3차원 형태인 복셀 데이터로 변환시킵니다.

메쉬를 입력받아 정규화 과정을 거친 후 복셀 배열을 생성합니다.

삼각형 면을 감싸는 영역의 복셀만 검사하여 삼각형이 차지하는 영역에 표시합니다.

**Classifier.cs**

학습된 ONNX모델을 가져와 사용하는 클래스입니다.

전달받은 메쉬의 데이터를 전처리하여 복셀화하여 학습한 모델이 사용할 수 있게 변환시켜줍니다.

ONNX모델을 구동하여 물체의 종류를 판별하고 결과를 출력합니다.
