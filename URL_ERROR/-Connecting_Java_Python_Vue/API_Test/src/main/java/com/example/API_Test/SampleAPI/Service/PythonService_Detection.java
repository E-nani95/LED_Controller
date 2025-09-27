package com.example.API_Test.SampleAPI.Service;

import com.example.API_Test.SampleAPI.Controller.DTO.Request_Detection_Python;
import com.example.API_Test.SampleAPI.Controller.DTO.Response_Detection_Python;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class PythonService_Detection {
    private final WebClient webClient;
//    public Response_Python Python_API(Request_Python requestPython) {
//        return webClient.post()
//                .uri("/APITest") // 기본 URL 뒤에 붙는 경로
//                .bodyValue(requestPython) // 요청 본문에 item 객체를 JSON으로 넣어 보냄
//                .retrieve()  // 응답을 받아옴
//                .bodyToMono(Response_Python.class) // 응답 본문을 AnalysisResult DTO로 변환
//                .block(); // 비동기 결과를 동기적으로 기다림 (간단한 예제를 위해 사용)
//    }
public Response_Detection_Python Python_API(String userchoice, String path) {
//    int userChoiceInt = Integer.parseInt(String.valueOf(userchoice));
    Request_Detection_Python RR= new Request_Detection_Python(userchoice,path);
    return webClient.post()
            .uri("/APITest") // 기본 URL 뒤에 붙는 경로
            .bodyValue(RR) // 요청 본문에 item 객체를 JSON으로 넣어 보냄
            .retrieve()  // 응답을 받아옴
            .bodyToMono(Response_Detection_Python.class) // 응답 본문을 AnalysisResult DTO로 변환
            .block(); // 비동기 결과를 동기적으로 기다림 (간단한 예제를 위해 사용)
}
}
