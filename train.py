import numpy as np

import tensorflow as tf

import os

import tf2onnx

import random

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Dropout

from tensorflow.keras.models import Model





DATA_PATH = "datatxt" # 데이터를 가져올 위치

RESOLUTION = 32 # 데이터 크기 추출한 크기에 맞춰서 설정하기

INPUT_SHAPE = (RESOLUTION, RESOLUTION, RESOLUTION, 1) 



# 데이터 로딩 추출한 데이터를 읽어와 배열로 변환함

def load_data(data_path):

    all_voxels, all_labels = [], []

    # 디렉토리 이름으로 클래스 목록 추출
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    label_map = {name: i for i, name in enumerate(class_names)}

    print(f"클래스 발견: {class_names}")



    # 클래스별 폴더를 순회하며 데이터 로드
    for class_name, label_idx in label_map.items():

        class_path = os.path.join(data_path, class_name)

        for file_name in os.listdir(class_path):

            if file_name.endswith(".txt"):

                file_path = os.path.join(class_path, file_name)

                with open(file_path, 'r') as f:

                    lines = f.readlines()



                #첫 번째 줄의 해상도 정보
                res_x, res_y, res_z = map(int, lines[0].split())

                voxel_grid = np.zeros((res_x, res_y, res_z), dtype=np.uint8)

                #두 번째 줄 부터 좌표
                for line in lines[1:]:

                    x, y, z = map(int, line.split())

                    voxel_grid[x, y, z] = 1



                all_voxels.append(voxel_grid)

                all_labels.append(label_idx)



    # CNN 입력을 위해 규격에 맞게 가공
    # [데이터 개수, 32, 32, 32] -> [데이터 개수, 32, 32, 32, 1] (채널 추가)
    X = np.array(all_voxels, dtype=np.float32)[..., np.newaxis] 
    # 라벨데이터를 Numpy 배열로 변환
    y = np.array(all_labels, dtype=np.int32)

    return X, y, class_names





# 데이터 증강함수

def random_augment(voxel):

    # 각 축 독립 회전 (0, 90, 180, 270 중 하나) 

    # 0,1 = Z축과 Y축이 만드는 평면 기준 , 0,2 Z축과 X 축이 만드는 평면 기준 , 1,2 X축과 Y축이 만나는 평면 기준
    for axes in [(0, 1), (0, 2), (1, 2)]:

        k = np.random.randint(0, 4)

        if k: 

            voxel = np.rot90(voxel, k, axes=axes)



    # 무작위 반전

    for axis in range(3):

        if np.random.rand() < 0.5:

            voxel = np.flip(voxel, axis=axis)



    # 노이즈 추가 
    if np.random.rand() < 0.3:

        noise = np.random.normal(0, 0.05, voxel.shape)

        voxel = np.clip(voxel + noise, 0, 1)



    return voxel.astype(np.float32)







# 데이터 로드 및 전처리

print("1. 데이터 로딩...")

X, y, class_names = load_data(DATA_PATH)

num_classes = len(class_names)

print(f"로딩 완료! 총 {len(X)}개 데이터, {num_classes}개 클래스")


# 학습 및 검증 데이터 분할
# test_size = 0.2 : 전체 데이터에서 20%를 평가용으로 사용하여 과적합 방지
# stratify=y: 클래스 불균형 방지 원본과 동일한 비율로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, stratify=y, random_state=42

)



# tf.data 파이프라인 구성 (증강 포함) 

def augment_wrapper(x, y):

    #NumPy 연산을 TensorFlow 데이터 파이프라인에 통합
    x = tf.numpy_function(random_augment, [x], tf.float32)

    x.set_shape(INPUT_SHAPE)  # TensorFlow에게 shape 알려주기

    return x, y





#학습용 데이터셋 구성

train_dataset = (

    tf.data.Dataset.from_tensor_slices((X_train, y_train))

    .shuffle(1000)

    .map(augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    .batch(16)

    .prefetch(tf.data.AUTOTUNE)

)

# 검증용 데이터셋 구성

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)



# 모델 생성 

print("\n2. 3D CNN 모델 생성...")



inputs = Input(shape=INPUT_SHAPE, name="input_1")

# 3D 필터를 사용하여 공간적 패턴 추출
# strides=2 : 차원 축소하여 효율 높임
x = Conv3D(32, kernel_size=3, activation="relu", padding="same", strides=2)(inputs)

x = Conv3D(64, kernel_size=3, activation="relu", padding="same", strides=2)(x)

#  Flatten: 추출한 3D 특징 맵을 1차원 벡터로 변환
x = Flatten()(x)

# Dense & Dropout : 추출된 특징을 바탕으로 최종 분류 수행
x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

# Output: 클래스 개수만큼의 확률값을 출력하는 출력층
outputs = Dense(num_classes, activation="softmax", name="output_1")(x)

model = Model(inputs=inputs, outputs=outputs)



# 모델설정 
# adam 학습 방향과 속도를 동시에 조절하여 효율적으로 최적화하는 알고리즘
# 라벨값을 정수로 나타내기 위해 sparse_categorical_crossentropy 다중 분류 손실 함수를 사용
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()



#  학습 

print("\n3. 모델 학습 시작...")

# 전체 데이터셋을 60회 반복 학습시켜 최적의 가중치 탐색
model.fit(train_dataset, validation_data=test_dataset, epochs=60)



# 유니티 센티스에 사용할 수 있도록 ONNX 모델로 변환

print("\n4. ONNX 모델로 변환 및 저장...")

spec = (tf.TensorSpec((None, *INPUT_SHAPE), tf.float32, name="input_1"),)

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)



with open("My3DClassifier_Augmented.onnx", "wb") as f:

    f.write(onnx_model.SerializeToString())

print("✅ ONNX 모델 저장 완료: My3DClassifier_Augmented.onnx")



with open("labels.txt", "w") as f:

    f.write("\n".join(class_names))

print("✅ 라벨 파일 저장 완료: labels.txt")

