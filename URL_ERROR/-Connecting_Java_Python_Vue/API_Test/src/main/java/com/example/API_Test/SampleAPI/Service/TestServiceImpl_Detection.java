package com.example.API_Test.SampleAPI.Service;

import com.example.API_Test.SampleAPI.Controller.DTO.Request_Detection_Front;
import com.example.API_Test.SampleAPI.Repository.Entity.TestEntity;
import com.example.API_Test.SampleAPI.Repository.TestRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class TestServiceImpl_Detection implements TestService {
    private final TestRepository testRepository;
        @Override
        public String API_Front_Test(String both, String path) {
            try {
                TestEntity entity = TestEntity.builder()
                        .userChoice(both)
                        .ImageSavePath(path)
                        .build();
                testRepository.save(entity);
                return "success";
            }catch (Exception e){
                return "TestServiceError";
            }



    }
}
