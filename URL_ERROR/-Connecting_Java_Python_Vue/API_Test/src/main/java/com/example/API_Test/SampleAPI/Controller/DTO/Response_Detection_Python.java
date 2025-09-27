package com.example.API_Test.SampleAPI.Controller.DTO;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Response_Detection_Python {

    private String result;
    private double probability;

//    private String resultString;
//    private boolean resultBoolen;
}
