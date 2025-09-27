package com.example.API_Test.SampleAPI.Service;

import com.example.API_Test.SampleAPI.Controller.DTO.Request_Detection_Front;
import com.example.API_Test.SampleAPI.Controller.DTO.Request_Detection_Python;
import com.example.API_Test.SampleAPI.Controller.DTO.Request_Prompt_Python;
import com.example.API_Test.SampleAPI.Controller.DTO.Response_Prompt_Python;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class PythonService_Prompt {
    private final WebClient webClient;
    public Response_Prompt_Python Python_Prompt(String account,String prompt) {
        Request_Prompt_Python RP = new Request_Prompt_Python(account,prompt);
        return webClient.post()
                .uri("/CreateImage")
                .bodyValue(RP)
                .retrieve()
                .bodyToMono(Response_Prompt_Python.class)
                .block();

    }
}
