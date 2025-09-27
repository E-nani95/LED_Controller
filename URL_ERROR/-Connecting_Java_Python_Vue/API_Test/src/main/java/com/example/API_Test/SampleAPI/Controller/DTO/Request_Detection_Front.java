package com.example.API_Test.SampleAPI.Controller.DTO;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
@AllArgsConstructor
public class Request_Detection_Front {

    private String userAccount;
    private MultipartFile data;

//    Test코드
//    private String name;
//    private String phoneNumber;


}
