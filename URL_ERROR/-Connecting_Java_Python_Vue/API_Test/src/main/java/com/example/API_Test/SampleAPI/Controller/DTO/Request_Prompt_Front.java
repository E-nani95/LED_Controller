package com.example.API_Test.SampleAPI.Controller.DTO;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Request_Prompt_Front {
    private String Account;
    private String Prompt;
}
