using UnityEngine;

using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;


public class Classifier : MonoBehaviour
{
    // 학습된 ONNX 모델을 가져와 사용하는 클래스
    [Header("AI 모델 설정")]
    public Unity.InferenceEngine.ModelAsset modelAsset; // ONNX 모델파일
    public TextAsset labelsAsset; // 분류 항목 이름이 적힌 텍스트

    private Unity.InferenceEngine.Worker worker; // 모델 실행을 담당하는 추론 엔진
    private Unity.InferenceEngine.Model runtimeModel;
    private string[] labels;
    // 입력 데이터 규격
    private readonly Unity.InferenceEngine.TensorShape inputShape = new Unity.InferenceEngine.TensorShape(1, 32, 32, 32, 1);

    // 중복 실행 방지 플래그
    private bool _isClassifying = false;


    async void Start() // async 추가
    {
        try
        {
            // 모델로드
            runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
            worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.GPUCompute);
            labels = labelsAsset.text.Split('\n')
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrEmpty(s))
                .ToArray();

            // 1. 첫 실행 시 자원을 많이 사용하므로, 빈 데이터를 미리 전송해서 안정화
            using var warmupTensor = new Unity.InferenceEngine.Tensor<float>(inputShape);
            worker.Schedule(warmupTensor);

            // 2. 출력 결과를 미리 한 번 확인
            var output = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

            if (output != null)
            {
                using var result = await output.ReadbackAndCloneAsync();
            }

            Debug.Log("Sentis Classifier 초기화 및 GPU 예열 완료.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"초기화 중 오류 발생: {e.Message}");
        }
    }

    // 가장 확률이 높은 결과를 반환
    public async Task<(string name, float confidence)> ClassifyAsync(Mesh targetMesh, Transform targetTransform)
    {
        if (_isClassifying)
        {
            Debug.LogWarning("이미 다른 분류 작업이 진행 중입니다.");
            return ("처리 중...", 0f);
        }

        if (targetMesh == null || !targetMesh.isReadable)
        {
            Debug.LogError("메쉬가 null이거나 읽기 권한이 없습니다.");
            return ("메쉬 분석 불가", 0f);
        }

        _isClassifying = true;

        try
        {
            // 데이터 전처리
            Vector3[] vertices = targetMesh.vertices;
            int[] triangles = targetMesh.triangles;
            Bounds bounds = targetMesh.bounds;
            
            // 회전값 보정
            Quaternion originalRotation = targetTransform.rotation;
			Quaternion alignToUp = Quaternion.FromToRotation(targetTransform.up, Vector3.up);
			targetTransform.rotation = alignToUp * targetTransform.rotation;
			targetTransform.rotation = Quaternion.identity;

            // 월드 좌표계 기준으로 모든 정점 위치 변환
            Matrix4x4 localToWorld = targetTransform.localToWorldMatrix;
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] = localToWorld.MultiplyPoint3x4(vertices[i]);


            // 원래 회전값으로 복구
            targetTransform.rotation = originalRotation;

            // 변환된 정점들을 기준으로 전체 크기 재계산
            bounds = RecalculateBounds(vertices);


			// 복셀화
            // 점과 삼각형 정보를 32*32*32 그리드 형태의 bool 배열로 변환
			bool[,,] voxelData = await Voxelizer.VoxelizeFromDataAsync(vertices, triangles, bounds);


			if (voxelData == null)
                return ("복셀 데이터 생성 실패", 0f);

            // 텐서 생성
            // 배열을 AI 모델이 사용할 수 있게 1차원 float 배열로 변환
            var data = new float[32 * 32 * 32];
            int index = 0;
            for (int z = 0; z < 32; z++)
                for (int y = 0; y < 32; y++)
                    for (int x = 0; x < 32; x++)
                        data[index++] = voxelData[x, y, z] ? 1f : 0f;

            // 모델 실행
            using var inputTensor = new Unity.InferenceEngine.Tensor<float>(inputShape, data);
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

            // GPU 작업이 끝날 때까지 메인 스레드를 멈추지 않고 비동기적으로 대기
            using var cpuTensor = await outputTensor.ReadbackAndCloneAsync();

            ProcessAllOutputs(cpuTensor);
            return ProcessOutput(cpuTensor);
        }
        finally
        {
            _isClassifying = false;
        }
    }

    // 모든 결과 분류
    public async Task<List<(string name, float confidence)>> ClassifyAsyncAll(Mesh targetMesh, Transform targetTransform)
    {
        if (_isClassifying)
        {
            Debug.LogWarning("이미 다른 분류 작업이 진행 중입니다.");
            return null;
        }

        if (targetMesh == null || !targetMesh.isReadable)
        {
            Debug.LogError("메쉬가 null이거나 읽기 권한이 없습니다.");
            return null;
        }

        _isClassifying = true;

        try
        {
            Vector3[] vertices = targetMesh.vertices;
            int[] triangles = targetMesh.triangles;
            Bounds bounds = targetMesh.bounds;

            Quaternion originalRotation = targetTransform.rotation;
            Quaternion alignToUp = Quaternion.FromToRotation(targetTransform.up, Vector3.up);
            targetTransform.rotation = alignToUp * targetTransform.rotation;


            targetTransform.rotation = Quaternion.identity;
            Matrix4x4 localToWorld = targetTransform.localToWorldMatrix;
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] = localToWorld.MultiplyPoint3x4(vertices[i]);


            targetTransform.rotation = originalRotation;

            bounds = RecalculateBounds(vertices);


            bool[,,] voxelData = await Voxelizer.VoxelizeFromDataAsync(vertices, triangles, bounds);


            if (voxelData == null)
                return null;

            var data = new float[32 * 32 * 32];
            int index = 0;
            for (int z = 0; z < 32; z++)
                for (int y = 0; y < 32; y++)
                    for (int x = 0; x < 32; x++)
                        data[index++] = voxelData[x, y, z] ? 1f : 0f;

            using var inputTensor = new Unity.InferenceEngine.Tensor<float>(inputShape, data);
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

            using var cpuTensor = await outputTensor.ReadbackAndCloneAsync();


            return ProcessAllOutputs(cpuTensor);
        }
        finally
        {
            _isClassifying = false;
        }
    }

    // 정점 배열을 순회하며 전체 메쉬를 감싸는 영역을 계산
    private static Bounds RecalculateBounds(Vector3[] vertices)
    {
        if (vertices == null || vertices.Length == 0)
            return new Bounds(Vector3.zero, Vector3.zero);

        Vector3 min = vertices[0], max = vertices[0];
        for (int i = 1; i < vertices.Length; i++)
        {
            min = Vector3.Min(min, vertices[i]);
            max = Vector3.Max(max, vertices[i]);
        }
        return new Bounds((min + max) * 0.5f, max - min);
    }

    // 출력 텐서에서 가장 높은 값을 찾아 해당 라벨 이름과 확률 반한
    private (string name, float confidence) ProcessOutput(Unity.InferenceEngine.Tensor<float> outputTensor)
    {
        if (outputTensor == null) return ("출력 텐서 없음", 0f);
        int length = outputTensor.count;
        float maxProb = float.MinValue;
        int predIdx = -1;
        for (int i = 0; i < length; i++)
        {
            float v = outputTensor[i];
            if (v > maxProb) { maxProb = v; predIdx = i; }
        }
        if (predIdx < 0 || predIdx >= labels.Length) return ("Unknown", 0f);
        return (labels[predIdx], maxProb);
    }

    // 전체 결과를 리스트화해서 확률이 높은 순으로 출력
    private List<(string name, float confidence)> ProcessAllOutputs(Unity.InferenceEngine.Tensor<float> outputTensor)
    {
        if (outputTensor == null)
        {
            Debug.LogError("ProcessAllOutputs: 출력 텐서가 null입니다.");
            return null;
        }

        // 모든 확률과 라벨을 저장할 리스트 생성
        var results = new List<(string name, float confidence)>();
        int length = Mathf.Min(outputTensor.count, labels.Length);

        for (int i = 0; i < length; i++)
        {
            float probability = outputTensor[i];
            results.Add((labels[i], probability));
        }

        // 확률이 높은 순으로 결과를 정렬
        var sortedResults = results.OrderByDescending(r => r.confidence).ToList();

        Debug.Log("--- Voxel Classification Results ---"); 
        foreach (var result in sortedResults)
        {
            Debug.Log($"[Category] {result.name}: {result.confidence:P2}"); // P2는 백분율(Percentage)로 소수점 두 자리까지 표시
        }
        Debug.Log("------------------------------------");

        return sortedResults;
    }
    void OnDestroy()
    {
        worker?.Dispose();
    }
}