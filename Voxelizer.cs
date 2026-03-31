using UnityEngine;
using System.Collections.Generic;
using System.Threading.Tasks;

public static class Voxelizer
{

	// 메쉬 데이터를 받아 복셀 배열로 변환

	public static bool[,,] VoxelizeMesh(Mesh mesh, int resolution = 32)
	{
		bool[,,] voxels = new bool[resolution, resolution, resolution];

		// 메쉬 경계 영역 계산
		mesh.RecalculateBounds();
		Bounds bounds = mesh.bounds;

		Vector3 min = bounds.min;
		Vector3 max = bounds.max;
		Vector3 size = max - min;

		//크기 정규화 가장 긴 축을 기준으로 0~1사이의 비율로 맞춤
		float maxSize = Mathf.Max(size.x, size.y, size.z);
		Vector3 scale = Vector3.one * (1f / maxSize);

		//중심 정규화 모델이 공간의 정중앙에 오도록 이동값 계산
		Vector3 center = (min + max) * 0.5f; // 원래 중심
		Vector3 offset = Vector3.one * 0.5f - Vector3.Scale(center - min, scale);

		Vector3[] vertices = mesh.vertices;
		int[] triangles = mesh.triangles;

		// 메쉬를 구성하는 모든 삼각형 면을 순회하며 복셀을 채움
		for (int i = 0; i < triangles.Length; i += 3)
		{
			// 정점 좌표를 0~1 사이의 정규화된 공간 좌표로 변환
			Vector3 v0 = Vector3.Scale(vertices[triangles[i]] - min, scale) + offset;
            Vector3 v1 = Vector3.Scale(vertices[triangles[i + 1]] - min, scale) + offset;
            Vector3 v2 = Vector3.Scale(vertices[triangles[i + 2]] - min, scale) + offset;

			// 해당 삼각형이 차지하는 영역에 복셀 표시
			FillTriangleVoxels(voxels, v0, v1, v2, resolution);
		}

		return voxels;
	}

	// 특정 삼각형 면이 지나가는 위치의 복셀을 찾아 True로 설정
	private static void FillTriangleVoxels(bool[,,] voxels, Vector3 v0, Vector3 v1, Vector3 v2, int res)
	{
		// 삼각형 하나를 감싸는 아주 작은 Bounds 생성
		Bounds triBounds = new Bounds(v0, Vector3.zero);
		triBounds.Encapsulate(v1);
		triBounds.Encapsulate(v2);

		int minX = Mathf.Clamp((int)(triBounds.min.x * res), 0, res - 1);
		int minY = Mathf.Clamp((int)(triBounds.min.y * res), 0, res - 1);
		int minZ = Mathf.Clamp((int)(triBounds.min.z * res), 0, res - 1);
		int maxX = Mathf.Clamp((int)(triBounds.max.x * res), 0, res - 1);
		int maxY = Mathf.Clamp((int)(triBounds.max.y * res), 0, res - 1);
		int maxZ = Mathf.Clamp((int)(triBounds.max.z * res), 0, res - 1);

		// 삼각형이 위치한 영역 내의 복셀들만 루프 수행
		for (int x = minX; x <= maxX; x++)
			for (int y = minY; y <= maxY; y++)
				for (int z = minZ; z <= maxZ; z++)
				{
					Vector3 p = new Vector3(
						(x + 0.5f) / res,
						(y + 0.5f) / res,
						(z + 0.5f) / res
					);

					if (PointNearTriangle(p, v0, v1, v2, 1f / res))
						voxels[x, y, z] = true;
				}
	}

	// 점 p가 삼각형 면으로부터 일정 거리 안에 있는지 확인
	private static bool PointNearTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c, float threshold)
	{
		Vector3 normal = Vector3.Cross(b - a, c - a).normalized;
		float dist = Mathf.Abs(Vector3.Dot(p - a, normal));
		return dist < threshold; // 삼각형 평면과 가까우면 채움
	}


	public static Task<bool[,,]> VoxelizeFromDataAsync(Vector3[] vertices, int[] triangles, Bounds bounds, int resolution = 32)
	{
		return Task.Run(() =>
		{
			// 데이터 추출 및 스케일링 설정
			bool[,,] voxels = new bool[resolution, resolution, resolution];
			Vector3 min = bounds.min;
			Vector3 max = bounds.max;
			Vector3 size = max - min;
			float maxSize = Mathf.Max(size.x, size.y, size.z);
			Vector3 scale = Vector3.one * (1f / maxSize);
			Vector3 center = (min + max) * 0.5f;
			Vector3 offset = Vector3.one * 0.5f - Vector3.Scale(center - min, scale);

			for (int i = 0; i < triangles.Length; i += 3)
			{
				Vector3 v0 = Vector3.Scale(vertices[triangles[i]] - min, scale) + offset;
				Vector3 v1 = Vector3.Scale(vertices[triangles[i + 1]] - min, scale) + offset;
				Vector3 v2 = Vector3.Scale(vertices[triangles[i + 2]] - min, scale) + offset;
				FillTriangleVoxels(voxels, v0, v1, v2, resolution);
			}
			return voxels;
		});
	}
}
