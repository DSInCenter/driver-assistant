package com.example.driverassistant;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    
    Button btnFaceRegonition;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        init();
        
        btnFaceRegonition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, FaceRecognitionCameraActivity.class);
                startActivity(intent);
            }
        });
    }
    
    private void init() {
        btnFaceRegonition = findViewById(R.id.btn_main_face_recognition);
    }
}