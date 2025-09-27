package com.example.API_Test.SampleAPI.Repository;

import com.example.API_Test.SampleAPI.Repository.Entity.TestEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TestRepository extends JpaRepository<TestEntity,Long> {
}
