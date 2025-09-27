package com.example.API_Test.SampleAPI.Controller;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class Python_API {

//    public WebClient webClient(WebClient.Builder webClientBuilder)
//    ()안에 안쓰면 application.properties에 적용한 설정값이 적용안되고 다 디폴트값으로 적용함

    @Bean
    public WebClient webClient(WebClient.Builder webClientBuilder) {
        return WebClient.builder().baseUrl("http://127.0.0.1:4999")
                .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(50 * 1024 * 1024))
                .build();
    }
}
