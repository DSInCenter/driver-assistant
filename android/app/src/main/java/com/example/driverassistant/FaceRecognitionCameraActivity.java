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
                    byte[] data = frame.getData();
                    inputImage = InputImage.fromByteArray(data, size.getWidth(), size.getHeight(),
                            userRotation, format);

                } else if (frame.getDataClass() == Image.class) {
                    Image data = frame.getData();
                    inputImage = InputImage.fromMediaImage(data, userRotation);
                }
                
                Task<List<Face>> result = detector.process(inputImage)
                        .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                            @Override
                            public void onSuccess(@NonNull List<Face> faces) {
                                // TODO: 8/3/21
                                for (Face face : faces) {
                                    RectF box = new RectF(face.getBoundingBox());

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