using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;



public class BatchModelprocessor
{
	// 유니티에서 모델들을 순회하며 학습용 데이터로 변환하는 도구

	[MenuItem("Tools/BatchModelprocessor/Process Training Folder")]
	public static void ProcessAllMeshes()
	{
		string targetFolder = "Assets/training"; // 원본 모델들 경로
		string outputRoot = "Assets/datatxt";  // 출력 폴더
		int resolution = 32; // 모델 입력 규격

		// 지정된 폴더 내의 모든 모델 ( Prefab 포함 )검색
		string[] guids = AssetDatabase.FindAssets("t:Model t:Mesh t:Prefab", new[] { targetFolder });
		if (guids.Length == 0)
		{
			Debug.LogWarning("No mesh/model/prefab files found in " + targetFolder);
			return;
		}

		if (!AssetDatabase.IsValidFolder(outputRoot))
			AssetDatabase.CreateFolder("Assets", "datatxt");

		// 검색된 모든 에셋에 대해 반복 처리
		foreach (string guid in guids)
		{
			string assetPath = AssetDatabase.GUIDToAssetPath(guid).Replace("\\", "/");
			Object asset = AssetDatabase.LoadAssetAtPath<Object>(assetPath);

			List<Mesh> meshes = ExtractMeshes(asset);
			if (meshes.Count == 0)
				continue;

			// 원본 폴더 구조 그대로 출력 폴더 구죠 유지
			string relativePath = Path.GetDirectoryName(assetPath).Replace("\\", "/");
			if (relativePath.StartsWith(targetFolder))
				relativePath = relativePath.Substring(targetFolder.Length).TrimStart('/');

			string outputFolder = Path.Combine(outputRoot, relativePath).Replace("\\", "/");
			CreateFolderRecursively(outputFolder);

			// 추출된 각 메쉬 복셀화 하고 파일로 저장
			foreach (Mesh mesh in meshes)
			{
				bool[,,] voxels = Voxelizer.VoxelizeMesh(mesh, resolution);
				SaveVoxelData(mesh.name, voxels, outputFolder);
			}
		}
		AssetDatabase.Refresh(); // 모든 저장 끝나고 한 번만
	}


	// 단일 메쉬 파일뿐만 아니라 Prefab내의 포함된 MeshFilter 에서도 메쉬를 추출
	private static List<Mesh> ExtractMeshes(Object asset)
	{
		List<Mesh> meshes = new List<Mesh>();

		if (asset is Mesh mesh)
		{
			meshes.Add(mesh);
		}
		else if (asset is GameObject go)
		{
			// MeshFilter 처리
			MeshFilter[] filters = go.GetComponentsInChildren<MeshFilter>(true);
			foreach (var f in filters)
				if (f.sharedMesh != null)
					meshes.Add(f.sharedMesh);

			// SkinnedMeshRenderer 처리
			SkinnedMeshRenderer[] skins = go.GetComponentsInChildren<SkinnedMeshRenderer>(true);
			foreach (var s in skins)
				if (s.sharedMesh != null)
					meshes.Add(s.sharedMesh);
		}
		return meshes;
	}

	// 복셀 데이터를 직렬화하여 저장함
	private static void SaveVoxelData(string name, bool[,,] voxels, string folder)
	{
        foreach (char c in Path.GetInvalidFileNameChars())
        {
            name = name.Replace(c, '_');
        }

        int res = voxels.GetLength(0);
		string fullFolder = Path.Combine(Application.dataPath, folder.Replace("Assets/", "")).Replace("\\", "/");

		if (!Directory.Exists(fullFolder))
			Directory.CreateDirectory(fullFolder);

		string filePath = Path.Combine(fullFolder, $"{name}_voxels_{res}.txt");

		using (StreamWriter writer = new StreamWriter(filePath))
		{
			writer.WriteLine($"{res} {res} {res}");
			for (int x = 0; x < res; x++)
				for (int y = 0; y < res; y++)
					for (int z = 0; z < res; z++)
						if (voxels[x, y, z])
							writer.WriteLine($"{x} {y} {z}");
		}
	}

	private static void CreateFolderRecursively(string folderPath)
	{
		string[] parts = folderPath.Split('/');
		string currentPath = parts[0]; // "Assets"
		for (int i = 1; i < parts.Length; i++)
		{
			string nextPath = Path.Combine(currentPath, parts[i]);
			if (!AssetDatabase.IsValidFolder(nextPath))
				AssetDatabase.CreateFolder(currentPath, parts[i]);
			currentPath = nextPath;
		}
	}
}
