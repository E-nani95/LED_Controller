package com.example.API_Test.SampleAPI.Controller.DTO;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

@Data
@AllArgsConstructor
public class Response_Prompt_Front {
    private String generatedImageUrl;
}
