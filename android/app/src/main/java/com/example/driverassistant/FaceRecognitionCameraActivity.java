package com.example.driverassistant;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Rect;
import android.graphics.RectF;
import android.media.Image;
import android.os.Bundle;
import android.widget.TextView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;
import com.otaliastudios.cameraview.size.Size;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.List;

public class FaceRecognitionCameraActivity extends AppCompatActivity {

    private CameraView camera;
    private TextView txtDetected;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_recognition_camera);
        camera = findViewById(R.id.camera_face_recognition);
        camera.setLifecycleOwner(this);
        FaceDetector detector = setupDetector();
        txtDetected = findViewById(R.id.txt_face_detected);
        camera.addFrameProcessor(new FrameProcessor() {
            @Override
            public void process(@NonNull Frame frame) {
                long time = frame.getTime();
                Size size = frame.getSize();
                int format = frame.getFormat(); //NV21 = 17
                int userRotation = frame.getRotationToUser();
                int viewRotation = frame.getRotationToView();
                InputImage inputImage = null;
                if (frame.getDataClass() == byte[].class) {
                    final byte[] data = frame.getData();
                    inputImage = InputImage.fromByteArray(data, size.getWidth(), size.getHeight(),
                            userRotation, format);

                } else if (frame.getDataClass() == Image.class) {
                    final Image data = frame.getData();
                    inputImage = InputImage.fromMediaImage(data, userRotation);
                }
//                System.out.println(inputImage.getByteBuffer());
                InputImage finalInputImage = inputImage;
                Task<List<Face>> result = detector.process(inputImage)
                        .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                            @Override
                            public void onSuccess(@NonNull List<Face> faces) {
                                // TODO: 8/3/21
                                for (Face face : faces) {
                                    RectF box = new RectF(face.getBoundingBox());
                                    byte[] cropped = cropByteArray(finalInputImage.getByteBuffer().array(), finalInputImage.getWidth(),face.getBoundingBox());
//                                    Bitmap bitmapImage = BitmapFactory.decodeByteArray(cropped, 0, cropped.length);
//                                    Bitmap mutableBitmapImage = Bitmap.createScaledBitmap(bitmapImage, 112, 112, false);
//                                    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
//                                    mutableBitmapImage.compress(Bitmap)
                                    try {
                                        Mat mat = Imgcodecs.imdecode(new MatOfByte(cropped), Imgcodecs.IMREAD_UNCHANGED);
                                        Mat resized = new Mat();
                                        org.opencv.core.Size sz = new org.opencv.core.Size(112, 112);
                                        Imgproc.resize(mat, resized, sz);
                                        txtDetected.setText(String.valueOf(time));
                                    } catch (Exception e){
                                        e.printStackTrace();
                                    }








                                }
                            }
                        }).addOnFailureListener(new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // TODO: 8/3/21  
                            }
                        });
                
            }
        });
    }

    private byte[] croppedNV21(Image mediaImage, Rect cropRect){
        ByteBuffer yBuffer = mediaImage.getPlanes()[0].getBuffer();
        ByteBuffer vuBuffer = mediaImage.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int vuSize = vuBuffer.remaining();

        byte[] nv21 = new byte[ySize + vuSize];
        yBuffer.get(nv21, 0, ySize);
        vuBuffer.get(nv21, ySize, vuSize);

        return cropByteArray(nv21, mediaImage.getWidth(), cropRect);

    }
//
    private byte[] cropByteArray(byte[] array, int imageWidth, Rect cropRect){
        byte[] croppedArray = new byte[(cropRect.width() * cropRect.height())];
        int i = 0;
        for (int index = 0; i < array.length; i++) {
            double x = index % imageWidth;
            double y = index / imageWidth;
                    if (cropRect.left <= x && x < cropRect.right && cropRect.top <= y && y < cropRect.bottom) {
                        croppedArray[i] = array[index];
                        i++;
                    }
        }
//        array.forEachIndexed { index, byte ->
//            val x = index % imageWidth
//            val y = index / imageWidth
//
//            if (cropRect.left <= x && x < cropRect.right && cropRect.top <= y && y < cropRect.bottom) {
//                croppedArray[i] = byte
//                i++
//            }
//        }

        return croppedArray;
    }

    private FaceDetector setupDetector() {
        FaceDetectorOptions detectorOptions = new FaceDetectorOptions.Builder()
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .build();
        return FaceDetection.getClient(detectorOptions);

    }

}