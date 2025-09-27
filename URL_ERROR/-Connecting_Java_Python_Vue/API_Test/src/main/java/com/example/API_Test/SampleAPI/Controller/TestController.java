package com.example.API_Test.SampleAPI.Controller;


import com.example.API_Test.SampleAPI.Controller.DTO.*;
import com.example.API_Test.SampleAPI.Service.ImageService;
import com.example.API_Test.SampleAPI.Service.PythonService_Detection;
import com.example.API_Test.SampleAPI.Service.PythonService_Prompt;
import com.example.API_Test.SampleAPI.Service.TestService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import javax.swing.*;
import java.io.IOException;

@RestController
@RequiredArgsConstructor
// origins에 프론트엔드 주소를 정확히 명시
@CrossOrigin(origins = {"http://localhost:3000","https://vue-web-8o3l.vercel.app"})
public class TestController {
    private final TestService testService;
    private final PythonService_Detection pythonServiceDetection;
    private final  ImageService imageService;
    private final PythonService_Prompt pythonServicePrompt;

    @GetMapping("/")
    public String IsIndex(){
        return "IndexPage";
    }

    @PostMapping("/TestAPI")
//    public Response_Front IsTest(@RequestBody Request_Front request_front){
    public Response_Detection_Front IsTest(@ModelAttribute Request_Detection_Front request_Detection_front) throws IOException {
            String path = imageService.saveImage(request_Detection_front.getData());
            System.out.println("Start");
            String result = testService.API_Front_Test(String.valueOf(request_Detection_front.getUserAccount()),path);
            System.out.println(result);
            System.out.println(request_Detection_front.getUserAccount());
            Response_Detection_Python Python_result = pythonServiceDetection.Python_API(String.valueOf(request_Detection_front.getUserAccount()),path);
            System.out.println(Python_result);
            if("success".equalsIgnoreCase(result)){
    //            return new Response_Front(1 ,"Good");
                return new Response_Detection_Front(Python_result.getResult() ,Python_result.getProbability());
    //            return new Response_Front("Test","Good",request_front.getSort());
            }else{
    //            return new Response_Front(2,"Bad");
                return new Response_Detection_Front(Python_result.getResult() ,Python_result.getProbability());
    //            return new Response_Front("Test","Bad",request_front.getName());
            }
    }

    @PostMapping("/PromptAPI")
    public Response_Prompt_Front IsPrompt(@RequestBody Request_Prompt_Front requestPromptFront ){
        System.out.println("check1");
        String result =testService.API_Front_Test(String.valueOf(requestPromptFront.getAccount()),String.valueOf(requestPromptFront.getPrompt()));
        System.out.println("check2");
        System.out.println(requestPromptFront.getAccount());
        System.out.println(requestPromptFront.getPrompt());
        Response_Prompt_Python responsePromptPython = pythonServicePrompt.Python_Prompt(requestPromptFront.getAccount(),requestPromptFront.getPrompt());
        System.out.println("check3");
//        System.out.println(responsePromptPython.getPath());
        if("success".equalsIgnoreCase(result)){
            System.out.println("Final");
            return new Response_Prompt_Front(String.valueOf(responsePromptPython.getPath()));
        }else{
            return null;
        }
    }

}
