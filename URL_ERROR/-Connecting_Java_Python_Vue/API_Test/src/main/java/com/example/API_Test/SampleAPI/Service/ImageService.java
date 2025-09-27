package com.example.API_Test.SampleAPI.Service;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Service
public class ImageService {
    //application.properties에 설정된 경로
    //file.upload-dir="C:/Users/AI-00/Desktop/API_Python_Java_Front/API_Test/saveImage"

    public String saveImage(MultipartFile file) throws IOException {
//        System.out.println("test");
        // 1 이름가져오기
//        String file_name= file.getName();
        String file_name= file.getOriginalFilename();
        //2 경로
        String Dir_path="C:/Users/AI-00/Desktop/Portfolio_Web/API_Test/saveImage/";
//        String path = Dir_path+file_name+".jpg";
        String path = Dir_path+file_name;
        boolean isItThere = checkDuplication(path);
        if(isItThere){
            for(int i=0; i<5; i++){
                String TestName = splitName(file_name,i);
                path=null;
                path=Dir_path+TestName;
                boolean checkCheck = checkDuplication(path);
                if(!checkCheck){
                    break;
                }
            }
        }
        file.transferTo(new File(path));
        return path;
    }

    public boolean checkDuplication(String path){
        File check = new File(path);
        return check.exists();
    }

    public String splitName(String name, int num){
       String[] nameList = name.split("\\.");

       return nameList[0]+"("+num+")"+"."+nameList[1];

    }

}