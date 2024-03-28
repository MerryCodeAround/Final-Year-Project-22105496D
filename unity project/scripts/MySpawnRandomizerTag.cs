using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

public class MySpawnRandomizerTag : RandomizerTag
{
    public void SetSpawn(Vector3 displacement, Vector3 rotation, int index)
    {
		transform.localPosition += displacement;
		transform.localRotation *= Quaternion.Euler(rotation);
		for (int i = 0; i < transform.childCount; i++)
		{
			Transform child = transform.GetChild(i);
			if (index == 0) child.gameObject.SetActive(false); 
			else if (i == index - 1) child.gameObject.SetActive(true);
			else child.gameObject.SetActive(false);
		}
	}
}
