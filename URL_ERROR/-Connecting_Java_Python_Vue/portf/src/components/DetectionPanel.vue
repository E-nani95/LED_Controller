<template>
  <div class="panel">
    <h2>Deepfake Detection</h2>
    
    <div 
      :class="['image-preview-box', { 'is-dragover': isDragOver }]"
      @dragover.prevent="onDragOver"
      @dragleave.prevent="onDragLeave"
      @drop.prevent="onDrop"
    >
      <img v-if="uploadedImagePreview" :src="uploadedImagePreview" alt="업로드된 이미지"/>
      <div v-else class="placeholder">
        <span>탐지할 이미지를 여기에 드롭하거나 버튼을 눌러 첨부해주세요.</span>
      </div>
    </div>

    <div v-if="uploadedFile" class="file-info">
      <p><strong>파일명:</strong> {{ uploadedFile.name }}</p>
      <p><strong>파일 크기:</strong> {{ (uploadedFile.size / 1024 / 1024).toFixed(2) }} MB</p>
    </div>
    <div class="result-box">
      <strong>결과:</strong>
      <div v-if="isLoading" class="loading-indicator">
        <div class="spinner-small"></div>
        <span>분석 중입니다...</span>
      </div>
      <span v-else-if="deepfakeResult">{{ deepfakeResult }}&nbsp;<strong>확률:</strong>{{ probability }}</span>
      <span v-else>-</span>
    </div>

    <div class="actions">
      <input type="file" @change="handleFileUpload" accept="image/*" ref="fileInput" style="display: none;" />
      <button class="button-secondary" @click="triggerFileUpload">파일 첨부</button>
      <button @click="detectDeepfake" :disabled="!uploadedFile || isLoading">
        {{ isLoading ? '탐지 중...' : '탐지 실행' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';

const emit = defineEmits(['update:error']);

const uploadedFile = ref(null);
const uploadedImagePreview = ref(null);
const deepfakeResult = ref(null);
const probability = ref(null);
const isLoading = ref(false);
const isDragOver = ref(false);
const apiBaseUrl = 'https://9114c55c85ba.ngrok-free.app';

const MAX_FILE_SIZE = 1 * 1024 * 1024;

const fileInput = ref(null);

const triggerFileUpload = () => {
  fileInput.value.click();
};

// [적용 완료] 드래그 앤 드롭을 위한 함수들
const onDragOver = () => { isDragOver.value = true; };
const onDragLeave = () => { isDragOver.value = false; }; // 오타 수정
const onDrop = (event) => {
  isDragOver.value = false;
  const file = event.dataTransfer.files[0];
  
  if (!file || !file.type.startsWith('image/')) {
    alert("이미지 파일만 첨부할 수 있습니다.");
    return;
  }

  // 기존 handleFileUpload의 로직과 동일하게 처리
  uploadedFile.value = file;
  uploadedImagePreview.value = URL.createObjectURL(file);
  deepfakeResult.value = null;
  probability.value = null;
};

// 이 아래 함수들은 전혀 수정되지 않았습니다.
const handleFileUpload = (event) => {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFile.value = file;
  uploadedImagePreview.value = URL.createObjectURL(file);
  deepfakeResult.value = null;
  probability.value = null;
};

const detectDeepfake = async () => {
  if (!uploadedFile.value) {
    alert("탐지할 이미지 파일을 먼저 첨부해주세요.");
    emit('update:error', "탐지할 이미지 파일을 먼저 첨부해주세요.");
    deepfakeResult.value = '파일없음'
    return;
  }
  if(uploadedFile.value.size > MAX_FILE_SIZE) {
    alert("파일 크기가 제한을 초과합니다.");
    emit('update:error', "파일 크기가 제한을 초과합니다.");
    deepfakeResult.value = '파일크기 제한초과'
    return;
  }
  
  emit('update:error', null);
  deepfakeResult.value = null;
  probability.value = null;
  isLoading.value = true;

  const formData = new FormData();
  formData.append('userAccount', "TempAccount");
  formData.append('data', uploadedFile.value);

  try {
    const response = await axios.post(`${apiBaseUrl}/TestAPI`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    deepfakeResult.value = response.data.userAccount;
    let probValue = Math.ceil(response.data.data * 10000) / 100;
    probability.value = probValue + "%";

  } catch (err) {
    console.error("Deepfake 탐지 API 호출 오류:", err);
    emit('update:error', "연결상태를 확인해주세요!!");
    deepfakeResult.value = '탐지 실패';

  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
/* 기존 스타일은 그대로 유지됩니다. */
.image-preview-box {
  border: 2px dashed #ccc;
  border-radius: 10px;
  cursor: default; /* 드래그앤드롭 전용이므로 커서 모양 변경 */
}
.image-preview-box.is-dragover {
  border-color: #4a90e2;
  background-color: #f0f8ff;
}
/* ... 이하 기존 스타일 ... */
.result-box {
  text-align: center;
  min-height: 48px;
  display: flex;
  align-items: center;
  font-size: 16px;
}
.result-box strong {
  margin-right: 8px;
}
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--secondary-color);
  font-size: 15px;
}
.spinner-small {
  border: 3px solid #f0f2f5;
  border-top: 3px solid var(--secondary-color);
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>