using System;
using System.Diagnostics;
using System.Security.Cryptography;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[Serializable]
[AddRandomizerMenu("My Spawn Randomizer")]
public class MySpawnRandomizer : Randomizer
{
	public GameObject[] prefabs; // Array of the prefabs to spawn
	public Vector3 displacementLimit = new Vector3();
	public Vector3 rotationLimit = new Vector3();

	//protected override void OnUpdate()
	protected override void OnIterationStart()
	{
		int[] passengers = new int[12]; // determine spawn or not, and what model. 0 = none, 1-3 = different models
		int length = UnityEngine.Random.Range(0, 12); // randomize length 
		for (int i = 0; i < length; i++) // initialize index
			passengers[i] = (i % 2) + 1; // example: 1,2,3,1,2,3,1,0,0,0...
		
		for (int i = 11; i >= 0; i--) // shuffle array
		{
			int randomIndex = UnityEngine.Random.Range(0, i); // Randomize a number between 0 and i (so that the range decreases each time)
			int temp = passengers[i]; // Save the value of the current i, otherwise it'll overright when we swap the values
			passengers[i] = passengers[randomIndex]; // Swap the new and old values
			passengers[randomIndex] = temp;
		}

		var tags = tagManager.Query<MySpawnRandomizerTag>(); // get all possible passenger model		
		int tagCounter = 0;
		foreach (var tag in tags)
		{
			float randomXDisplacement = UnityEngine.Random.Range(displacementLimit.x, -displacementLimit.x);
			float randomYDisplacement = UnityEngine.Random.Range(displacementLimit.y, -displacementLimit.y);
			float randomZDisplacement = UnityEngine.Random.Range(displacementLimit.z, -displacementLimit.z);
			float randomXRotation = UnityEngine.Random.Range(rotationLimit.x, -rotationLimit.x);
			float randomYRotation = UnityEngine.Random.Range(rotationLimit.y, -rotationLimit.y);
			float randomZRotation = UnityEngine.Random.Range(rotationLimit.z, -rotationLimit.z);
			Vector3 randomDisplacement = new Vector3(randomXDisplacement, randomYDisplacement, randomZDisplacement);
			Vector3 randomRotation = new Vector3(randomXRotation, randomYRotation, randomZRotation);
			tag.SetSpawn(randomDisplacement, randomRotation, passengers[tagCounter]);

			tagCounter++;
		}
	}
}

/*[Serializable]
[AddRandomizerMenu("My Spawn Randomizer")]
public class MySpawnRandomizer : Randomizer
{
	public GameObject[] prefabs; // Array of the prefabs to spawn
	public Transform parentObject;
	public Vector3 displacementLimit = new Vector3();
	public Vector3 rotationLimit = new Vector3();

	//protected override void OnUpdate()
	protected override void OnIterationStart()
	{
		//var tags = tagManager.Query<MySpawnRandomizerTag>(); // get all possible passenger models
		int childCounter = 0;
		foreach (Transform child in parentObject)
		{
			GameObject childObject = child.gameObject;
			int randomIndex = UnityEngine.Random.Range(0, prefabs.Length); // for random prefab model
			bool randomBool = UnityEngine.Random.Range(0, 2) > 0.9; // for spawn or not spawn
			float randomXDisplacement = UnityEngine.Random.Range(displacementLimit.x, -displacementLimit.x);
			float randomYDisplacement = UnityEngine.Random.Range(displacementLimit.y, -displacementLimit.y);
			float randomZDisplacement = UnityEngine.Random.Range(displacementLimit.z, -displacementLimit.z);
			float randomXRotation = UnityEngine.Random.Range(rotationLimit.x, -rotationLimit.x);
			float randomYRotation = UnityEngine.Random.Range(rotationLimit.y, -rotationLimit.y);
			float randomZRotation = UnityEngine.Random.Range(rotationLimit.z, -rotationLimit.z);
			Vector3 randomDisplacement = new Vector3(randomXDisplacement, randomYDisplacement, randomZDisplacement);
			Vector3 randomRotation = new Vector3(randomXRotation, randomYRotation, randomZRotation);
			GameObject prefabToSpawn = prefabs[randomIndex]; // Get the prefab at the random index

			// destroy previous prefab
			if (child.childCount == 1)
				UnityEngine.Object.Destroy(childObject.transform.GetChild(0).gameObject);

			// spawn a prefab
			if (randomBool == true)
			{
				GameObject spawnedObject = GameObject.Instantiate(prefabToSpawn, childObject.transform);
				spawnedObject.transform.localPosition += randomDisplacement;
				spawnedObject.transform.localRotation *= Quaternion.Euler(randomRotation);
			}

			//Perception.Labeling.RefreshLabeling();
			//RefreshLabeling();
			childCounter++;
		}
	}
}*/



/*public class MySpawnRandomizer : Randomizer
{
    public GameObject[] prefabs; // Array of the prefabs to spawn
    public Transform parentObject;

    //protected override void OnUpdate()
    protected override void OnIterationStart()
    {
        var tags = tagManager.Query<MySpawnRandomizerTag>(); // get all possible passenger models
        int tagCounter = 0;
        foreach (var tag in tags)
        {
            Transform parentObject = tag.GetComponent<Transform>();
            int randomIndex = UnityEngine.Random.Range(0, prefabs.Length); // for random prefab model
            bool randomBool = UnityEngine.Random.Range(0, 2) == 0; // for spawn or not spawn
            GameObject prefabToSpawn = prefabs[randomIndex]; // Get the prefab at the random index

            // destroy child
			if (parentObject.childCount == 1)
			{
				Transform childTransform = parentObject.GetChild(0); // Assuming the child is the first child
				GameObject childObject = childTransform.gameObject;
				UnityEngine.Object.Destroy(childObject);
				
			}

            // spawn a prefab
			if (randomBool == true)
            {
                //GameObject spawnedObject = GameObject.Instantiate(prefabToSpawn, parentObject);
                Debug.Log("spawn number " + randomIndex);
                GameObject.Instantiate(prefabToSpawn, parentObject);
            }

            tagCounter++;
        }
    }
}*/