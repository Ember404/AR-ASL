using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe;
using Mediapipe.Tasks.Vision.HandLandmarker;
using Mediapipe.Unity.Sample;
using UnityEngine.UI;
using System.Threading.Tasks;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Unity;
using Mediapipe.Unity.Experimental;
using Mediapipe.Unity.CoordinateSystem;
using System;
using PassthroughCameraSamples;
using System.IO;
using UnityEngine.Networking;
using Unity.Barracuda;
using System.Linq;
using Color = UnityEngine.Color;
using Unity.Collections;


public class HandLandmarks : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager m_webCamTextureManager;
    [SerializeField] private Text m_debugText;
    public RawImage myImage;

    [Header("Sign Model (Barracuda)")]
    [SerializeField] private NNModel signModelAsset;
    [SerializeField] private Text m_signDebugText;

    private Texture2D inputTexture;

    private HandLandmarker handLandmarker;
    private HandLandmarkerOptions options;
    private HandLandmarkerResult result;

    private Model signModel;
    private IWorker signWorker;
    private readonly string[] letterLabels = new[] {
        "A","B","C","D","E","F","G","H","I","K",
        "L","M","N","O","P","Q","R","S","T","U",
        "V","W","X","Y"
    };

    private IEnumerator Start()
    {
        while (m_webCamTextureManager.WebCamTexture == null)
            yield return null;

        myImage.texture = m_webCamTextureManager.WebCamTexture;
        var webTex = m_webCamTextureManager.WebCamTexture;
        inputTexture = new Texture2D(webTex.width, webTex.height, TextureFormat.RGBA32, false);

        string filePath = Path.Combine(Application.streamingAssetsPath, "hand_landmarker.task");
        UnityWebRequest request = UnityWebRequest.Get(filePath);
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success)
        {
            m_debugText.text = "failed to load file";
            yield break;
        }
        var modelData = request.downloadHandler.data;

        options = new HandLandmarkerOptions(
            baseOptions: new Mediapipe.Tasks.Core.BaseOptions(
                Mediapipe.Tasks.Core.BaseOptions.Delegate.CPU,
                modelAssetBuffer: modelData
            ),
            runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.IMAGE,
            numHands: 1,
            minHandDetectionConfidence: 0.5f,
            minHandPresenceConfidence: 0.5f,
            minTrackingConfidence: 0.5f
        );
        handLandmarker = HandLandmarker.CreateFromOptions(options, null);
        result = HandLandmarkerResult.Alloc(options.numHands);
        signModel = ModelLoader.Load(signModelAsset);
        signWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, signModel);

        DetectAndDraw();
    }

    private void Update() => DetectAndDraw();

    private Texture2D ConvertToGrayscale(Texture2D source)
    {
        NativeArray<byte> bytes = source.GetRawTextureData<byte>();
        for (int i = 0; i < bytes.Length; i += 4)
        {
            //byte gray = (byte)((bytes[i] * 0.299f) + (bytes[i + 1] * 0.587f) + (bytes[i + 2] * 0.114f));
            byte gray = (byte)((bytes[i] * 0.33f) + (bytes[i + 1] * 0.33f) + (bytes[i + 2] * 0.33f));

            bytes[i] = bytes[i + 1] = bytes[i + 2] = gray;
        }
        source.Apply();
        return source;
    }

    private Texture2D ResizeTexture(Texture2D sourceTexture, int targetWidth = 28, int targetHeight = 28)
    {
        // Tworzymy nową teksturę o docelowych wymiarach
        Texture2D resizedTexture = new Texture2D(targetWidth, targetHeight, sourceTexture.format, false);

        // Pobieramy oryginalne piksele
        UnityEngine.Color[] originalPixels = sourceTexture.GetPixels();
        UnityEngine.Color[] resizedPixels = new UnityEngine.Color[targetWidth * targetHeight];

        // Współczynniki skalowania
        float ratioX = 1.0f / (float)(targetWidth / (float)sourceTexture.width);
        float ratioY = 1.0f / (float)(targetHeight / (float)sourceTexture.height);

        // Przetwarzamy każdy piksel w docelowej teksturze
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Obliczamy odpowiedni piksel z oryginalnej tekstury
                int srcX = Mathf.FloorToInt(x * ratioX);
                int srcY = Mathf.FloorToInt(y * ratioY);
                int srcIndex = srcY * sourceTexture.width + srcX;

                // Kopiujemy piksel
                resizedPixels[y * targetWidth + x] = originalPixels[srcIndex];
            }
        }

        // Ustawiamy piksele dla nowej tekstury
        resizedTexture.SetPixels(resizedPixels);
        resizedTexture.Apply();

        return resizedTexture;
    }

    private void DetectAndDraw()
    {
        var webTex = m_webCamTextureManager.WebCamTexture;
        if (!webTex.isPlaying || webTex.width <= 16)
        {
            m_debugText.text = "Camera inactive";
            return;
        }

        inputTexture.SetPixels32(webTex.GetPixels32());
        inputTexture.Apply();

        if (inputTexture == null || inputTexture.width <= 16 || inputTexture.height <= 16)
        {
            m_debugText.text = "Invalid texture size";
            return;
        }

        using (var frame = new TextureFrame(inputTexture.width, inputTexture.height, TextureFormat.RGBA32))
        {
            frame.ReadTextureOnCPU(inputTexture, flipHorizontally: false, flipVertically: true);
            var mediaPipeImage = frame.BuildCPUImage();



            var imageOptions = new ImageProcessingOptions(rotationDegrees: 0);
            var result = HandLandmarkerResult.Alloc(options.numHands);


            bool success = handLandmarker.TryDetect(mediaPipeImage, imageOptions, ref result);
            if (!success || result.handLandmarks.Count == 0)
            {
                Debug.Log("nie wykryto dłoni");
                m_debugText.text = "No hands detected";
                return;
            }
            m_debugText.text = "Detected hand";

            var lmList = result.handLandmarks[0].landmarks;

            //int padding = 80;


            foreach (Transform child in myImage.transform) Destroy(child.gameObject);

            var rt = myImage.rectTransform.rect;
            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;

            foreach (var lm in result.handLandmarks[0].landmarks)
            {
                var pos = rt.GetPoint(in lm);
                pos.z = 0;

                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.transform.SetParent(myImage.transform, worldPositionStays: false);
                sphere.transform.localScale = Vector3.one * 5f;
                sphere.transform.localPosition = pos;

                //var mat = Resources.Load<Material>("Materials/red");
                //if (mat != null) sphere.GetComponent<Renderer>().material = mat;
            }


            minX = lmList.Min(lm => lm.x);
            maxX = lmList.Max(lm => lm.x);
            minY = lmList.Min(lm => lm.y);
            maxY = lmList.Max(lm => lm.y);

            int imgW = inputTexture.width;
            int imgH = inputTexture.height;

            //int x = Mathf.Clamp((int)(minX * imgW) - padding, 1, imgW);
            //int y = Mathf.Clamp((int)((1 - maxY) * imgH) - padding, 1, imgH);
            //int w = Mathf.Clamp((int)((maxX - minX) * imgW) + 2 * padding, 1, imgW);
            //int h = Mathf.Clamp((int)((maxY - minY) * imgH) + 2 * padding, 1, imgH);

            var x = Mathf.Clamp((int)(minX * imgW), 1, imgW);
            var y = Mathf.Clamp((int)((1 - maxY) * imgH), 1, imgH);
            var w = Mathf.Clamp((int)((maxX - minX) * imgW), 1, imgW);
            var h = Mathf.Clamp((int)((maxY - minY) * imgH), 1, imgH);

            // === Wytnij prostokąt ===
            Texture2D cropped = new Texture2D(w, h, TextureFormat.RGBA32, false);
            m_signDebugText.text = "sdfsd";

            UnityEngine.Color[] pixels = inputTexture.GetPixels(x, y, w, h);
            m_signDebugText.text = $"Cropped region: ({x}, {y}), Size: {w}x{h}";

            cropped.SetPixels(pixels);
            cropped.Apply();

            // === Konwersja do skali szarości ===
            Texture2D grayImage = ConvertToGrayscale(cropped);
            Texture2D rezisedImage = ResizeTexture(grayImage);

            float[] tensorData = new float[28 * 28];
            for (int i = 0; i < tensorData.Length; i++)
            {
                int pixelX = i % 28;
                int pixelY = 27 - (i / 28);
                UnityEngine.Color c = rezisedImage.GetPixel(pixelX, pixelY);
                tensorData[i] = c.r;
            }
            float minVal = tensorData.Min();
            float maxVal = tensorData.Max();
            string debugInfo = $"Input tensor shape: (1,28,28,1)\n" +
                       $"Min: {minVal:0.000}, Max: {maxVal:0.000}\n" +
                       $"tensorData:\n[{string.Join(",", tensorData.Select(f => f.ToString("0.000")))}]\n";
            Debug.Log(debugInfo);
            // === Tworzenie tensora i wykonanie predykcji ===
            using var inputTensor = new Tensor(1, 28, 28, 1, tensorData);
            signWorker.Execute(inputTensor);

            // === Pobranie wyników ===
            Tensor outputTensor = signWorker.PeekOutput();
            float[] scores = outputTensor.ToReadOnlyArray();

            // === Wyświetlenie top 3 wyników ===
            var top3 = scores
                .Select((val, idx) => new { Index = idx, Score = val })
                .OrderByDescending(x => x.Score)
                .Take(3)
                .ToList();

            string predictionResult = "";
            for (int i = 0; i < top3.Count; i++)
            {
                predictionResult += $"{i + 1}: {letterLabels[top3[i].Index]} ({top3[i].Score * 100f:0.0}%), ";
            }
            Debug.Log(predictionResult);
            //m_debugText.text = scores.ToString();
            m_signDebugText.text = predictionResult;
            inputTensor.Dispose();
            Destroy(cropped);
            Destroy(grayImage);
            Destroy(rezisedImage);
            frame.Dispose();
            mediaPipeImage.Dispose();
        }
        




        //using var frame = new TextureFrame(inputTexture.width, inputTexture.height, TextureFormat.RGBA32);
        //frame.ReadTextureOnCPU(inputTexture, false, true);
        //var mpImage = frame.BuildCPUImage();
        //var imgOpts = new ImageProcessingOptions(rotationDegrees: 0);

        //if (!handLandmarker.TryDetect(mpImage, imgOpts, ref result) || result.handLandmarks.Count == 0)
        //    return;

        //var lmList = result.handLandmarks[0].landmarks;
        //var rt = myImage.rectTransform;
        //float minX = float.MaxValue, minY = float.MaxValue;
        //float maxX = float.MinValue, maxY = float.MinValue;
        //foreach (var lm in lmList)
        //{
        //    Vector2 p = rt.rect.GetPoint(in lm);
        //    minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x);
        //    minY = Mathf.Min(minY, p.y); maxY = Mathf.Max(maxY, p.y);
        //}

        //float margin = 20f;
        //minX -= margin; minY -= margin;
        //maxX += margin; maxY += margin;

        //float texW = inputTexture.width, texH = inputTexture.height;
        //int x0 = Mathf.Clamp(Mathf.RoundToInt((minX / rt.rect.width) * texW), 0, (int)texW - 1);
        //int y0 = Mathf.Clamp(Mathf.RoundToInt(((rt.rect.height - maxY) / rt.rect.height) * texH), 0, (int)texH - 1);
        //int w = Mathf.Clamp(Mathf.RoundToInt(((maxX - minX) / rt.rect.width) * texW), 1, (int)texW - x0);
        //int h = Mathf.Clamp(Mathf.RoundToInt(((maxY - minY) / rt.rect.height) * texH), 1, (int)texH - y0);

        //// --- Create cropped texture and display it ---
        //var cropped = new Texture2D(w, h, TextureFormat.RGBA32, false);
        //var pixels = inputTexture.GetPixels(x0, y0, w, h);
        //cropped.SetPixels(pixels);
        //cropped.Apply();

        //// Show only the cropped hand
        ////myImage.texture = cropped;
        ////myImage.rectTransform.sizeDelta = new Vector2(w, h);

        //// (Optional) classification on cropped region
        //var tex28 = new Texture2D(28, 28, TextureFormat.R8, false);
        //for (int yy = 0; yy < 28; yy++)
        //    for (int xx = 0; xx < 28; xx++)
        //    {
        //        int srcX = Mathf.FloorToInt((xx / 28f) * w);
        //        int srcY = Mathf.FloorToInt((yy / 28f) * h);
        //        float gray = pixels[srcY * w + srcX].r + pixels[srcY * w + srcX].g + pixels[srcY * w + srcX].b;
        //        gray /= 3f; // Convert to grayscale
        //        tex28.SetPixel(xx, yy, new Color(gray, gray, gray));
        //    }
        //tex28.Apply();

        //myImage.texture = tex28;
        //myImage.rectTransform.sizeDelta = new Vector2(28, 28);

        //float[] tensorData = new float[28 * 28];
        //for (int i = 0; i < tensorData.Length; i++)
        //    tensorData[i] = tex28.GetPixel(i % 28, i / 28).r;

        //using var t = new Tensor(1, 28, 28, 1, tensorData);
        //signWorker.Execute(t);
        //var outTensor = signWorker.PeekOutput();
        //var scores = outTensor.ToReadOnlyArray();

        //var top5 = scores.Select((s, i) => (score: s, idx: i))
        //                  .OrderByDescending(p => p.score)
        //                  .Take(5)
        //                  .ToArray();

        //m_signDebugText.text = string.Join("\n", top5.Select((p, i) => $"{i + 1}: {letterLabels[p.idx]} {p.score * 100f:0.0}%"));
    }

    private void OnDestroy()
    {
        inputTexture = null;
    }
}
