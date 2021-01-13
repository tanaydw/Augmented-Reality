using System.Collections;
using System.Collections.Generic;
using SimpleJSON;
using UnityEngine;


public class GameObjectInfo : MonoBehaviour
{
    public int tracking_id;
    public int category;
    public int risk;
    public GameObject GO;

    public GameObjectInfo(GameObject GOBJ, int tid, int cat, float xc, float yc, float zc, int r)
    {
        tracking_id = tid;
        category = cat;
        risk = r;
        GO = Instantiate(GOBJ, new Vector3(xc, yc, zc), Quaternion.identity);
    }

    public void UpdateGameObjectInfo(float xc, float yc, float zc)
    {
        GO.transform.position = new Vector3(xc, yc, zc);
    }

    public void UpdateGameObjectPrefab(GameObject GOBJ, float xc, float yc, float zc, int r)
    {
        DestroyGameObject();
        risk = r;
        GO = Instantiate(GOBJ, new Vector3(xc, yc, zc), Quaternion.identity);
    }

    public void ChangeGameObjectPrefab(GameObject GOBJ, float xc, float yc, float zc)
    {
        DestroyGameObject();
        GO = Instantiate(GOBJ, new Vector3(xc, yc, zc), Quaternion.identity);
    }
    public void DestroyGameObject()
    {
        Destroy(GO);
    }
}


public class SpawnGameObjects : MonoBehaviour
{
    public GameObject Safe;
    public GameObject Warning;
    public GameObject Danger;
    public float Scale;
    public float Threshold;
    public string JsonPath;
    public string VideoPath;
    
    private int TotalFrames = 0;
    private string Json;
    private GameObject camera;
    private int Frame = 1;
    private JSONNode N;
    private float DangerDistance = 7.5F;
    private float WarningDistance = 15F;

    private List<GameObjectInfo> GameObjectInformation = new List<GameObjectInfo>();
    private List<int> TrackingIdentification = new List<int>();

    
    // Start is called before the first frame update
    void Start()
    {
        // Limiting the Frame rate for Synchronization
        Application.targetFrameRate = 30;

        // Processing JSON File
        Json = System.IO.File.ReadAllText(Application.dataPath + "/Data/" + JsonPath);
        N = JSON.Parse(Json);

        for (int i = 1; ; i++)
        {
            if (N[i.ToString()] != null)
            {
                TotalFrames += 1;
            }
            else
            {
                break;
            }
        }

        // Setting Up Video Camera
        camera = GameObject.Find("AR Camera");
        var videoPlayer = camera.AddComponent<UnityEngine.Video.VideoPlayer>();
        videoPlayer.playOnAwake = false;
        videoPlayer.targetCameraAlpha = 1.0F;
        videoPlayer.url = Application.dataPath + "/Data/" + VideoPath;
        videoPlayer.isLooping = true;
        videoPlayer.Play();
    }

    // Update is called once per frame
    void Update()
    {
        List<int> LocalTrackingIdentification = new List<int>();

        for(int i = 0; ; i++)
        {
            if (N[Frame.ToString()][i] != null)
            {
                if (N[Frame.ToString()][i]["score"] >= Threshold)
                {
                    float x = Scale * N[Frame.ToString()][i]["loc"][0].AsFloat;
                    float y = N[Frame.ToString()][i]["loc"][1].AsFloat;
                    float z = Scale * N[Frame.ToString()][i]["loc"][2].AsFloat;
                    float r = 0;

                    if (z <= DangerDistance * Scale)
                        r = 2;
                    else if (z > DangerDistance * Scale && z < WarningDistance * Scale)
                        r = 1;

                    int tid = N[Frame.ToString()][i]["tracking_id"];
                    int cat = N[Frame.ToString()][i]["class"];

                    LocalTrackingIdentification.Add(tid);

                    if (GameObjectInformation.Exists(t => t.tracking_id == tid))
                    {
                        int idx = GameObjectInformation.FindIndex(t => t.tracking_id == tid);

                        if (GameObjectInformation[idx].risk == r)
                            GameObjectInformation[idx].UpdateGameObjectInfo(x, y, z);
                        else
                        {
                            if (r == 0)
                                GameObjectInformation[idx].UpdateGameObjectPrefab(Safe, x, y, z, 0);
                            else if (r == 1)
                                GameObjectInformation[idx].UpdateGameObjectPrefab(Warning, x, y, z, 1);
                            else
                                GameObjectInformation[idx].UpdateGameObjectPrefab(Danger, x, y, z, 2);
                        }
                    }

                    else
                    {
                        if (z <= DangerDistance * Scale)
                            GameObjectInformation.Add(new GameObjectInfo(Danger, tid, cat, x, y, z, 2));
                        else if (z > DangerDistance * Scale && z < WarningDistance * Scale)
                            GameObjectInformation.Add(new GameObjectInfo(Warning, tid, cat, x, y, z, 1));
                        else
                            GameObjectInformation.Add(new GameObjectInfo(Safe, tid, cat, x, y, z, 0));

                        TrackingIdentification.Add(tid);
                    }
                }
            }

            else
            {
                for(int j = 0; j < TrackingIdentification.Count; j++)
                {
                    int tid = TrackingIdentification[j];

                    if (!LocalTrackingIdentification.Exists(t => t == tid))
                    {
                        int idx = GameObjectInformation.FindIndex(t => t.tracking_id == tid);
                        GameObjectInformation[idx].DestroyGameObject();
                        GameObjectInformation.RemoveAt(idx);
                        TrackingIdentification.RemoveAll(t => t == tid);
                    }
                }

                break;
            }
        }

        LocalTrackingIdentification.Clear();

        // Updating Frame
        Frame++;
        if (Frame == TotalFrames + 1)
        {
            Debug.Log("End");
            Frame = 1;
        }
        // Instantiate(SafeGameObject, new Vector3(Scale* SafeGameObject.distance* Mathf.Cos(SafeGameObject.alpha), -30, Scale* SafeGameObject.distance* Mathf.Sin(SafeGameObject.alpha));
        // Instantiate(Safe, new Vector3(52, -25, 500), Quaternion.identity);
        // Instantiate(Warning, new Vector3(-110, -25, 200), Quaternion.identity);
        // Instantiate(Danger, new Vector3(186, -25, 300), Quaternion.identity);
    }
}
