package com.example.API_Test.SampleAPI.Service;

import com.example.API_Test.SampleAPI.Controller.DTO.Request_Detection_Front;
import com.example.API_Test.SampleAPI.Repository.Entity.TestEntity;
import com.example.API_Test.SampleAPI.Repository.TestRepository;

public class TestServiceImpl_Prompt implements TestService{
    private TestRepository testRepository;
    @Override
    public String API_Front_Test(String both, String prompt) {
        try {
            TestEntity entity = TestEntity.builder()
                    .userChoice(both)
                    .ImageSavePath(prompt)
                    .build();
            testRepository.save(entity);
            return "success";
        }catch (Exception e){
            return "TestServiceError";
        }

    }
}
