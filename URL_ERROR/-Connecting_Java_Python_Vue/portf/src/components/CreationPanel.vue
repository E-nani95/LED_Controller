<template>
  <div class="panel">
    <h2>Create Image</h2>
    
    <div class="image-preview-box">
      <div v-if="isLoading" class="loader-container">
        <div class="loader"></div>
        <span>이미지를 생성 중입니다...</span>
      </div>
      <!-- <img v-else-if="generatedImageUrl" :src="generatedImageUrl" alt="생성된 이미지"/> -->
      <img v-else-if="generatedImageUrl" :src="generatedImageUrl" alt="생성된 이미지"/>
      <div v-else class="placeholder">
        <span>생성된 이미지가 여기에 표시됩니다.</span>
      </div>
    </div>
    
    <div class="command-input">
      <input 
        type="text" 
        v-model="imagePrompt" 
        placeholder="생성할 이미지에 대한 커맨드(프롬프트) 입력 ---> 영어로,  high quality 를 포함해주세요"
        @keyup.enter="generateImage"
        :disabled="isLoading"
      />
    </div>

    <div class="actions">
      <button @click="generateImage" :disabled="!imagePrompt || isLoading">
        {{ isLoading ? '생성 중...' : '생성' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';

// defineEmits를 사용하여 부모 컴포넌트로 이벤트를 보내는 함수를 정의합니다.
const emit = defineEmits(['update:error']);

// data() 속성을 ref를 사용한 반응형 변수로 변환합니다.
const imagePrompt = ref('');
const Account = ref(''); // 이 변수는 현재 코드에서 null로 초기화만 되고 사용되지 않지만, 구조를 그대로 유지했습니다.
const generatedImageUrl = ref('');
const isLoading = ref(false);
const apiBaseUrl = 'https://9114c55c85ba.ngrok-free.app';

// methods를 일반적인 const async 함수로 변환합니다.
const generateImage = async () => {
  if (!imagePrompt.value.trim()) {
    alert("이미지 생성 커맨드를 입력해주세요.");
    return;
  }

  // this.$emit 대신 emit 함수를 사용합니다.
  emit('update:error', null); // 부모의 에러 메시지 초기화
  
  // ref 변수의 값을 변경할 때는 .value를 사용합니다.
  isLoading.value = true;
  Account.value = null;
  generatedImageUrl.value = '';

  const payload = {
    Account: "TempPrompt",
    Prompt: imagePrompt.value,
  };

  try {
    const response = await axios.post(`${apiBaseUrl}/PromptAPI`, payload);
    console.log(response.data.generatedImageUrl);
    // 서버에서 받은 base64 문자열 앞에 Data URI Scheme을 붙여줍니다.
    generatedImageUrl.value = 'data:image/jpeg;base64,' + response.data.generatedImageUrl;
  } catch (err) {
    console.error("이미지 생성 API 호출 오류:", err);
    emit('update:error', "이미지 생성에 실패했습니다."); // 부모로 에러 전파
  } finally {
    isLoading.value = false;
  }
};
</script>


<!-- <script>
import axios from 'axios';

export default {
  name: 'CreationPanel',
  data() {
    return {
      imagePrompt: '',
      Account:'',
      generatedImageUrl: '',
      isLoading: false,
      // apiBaseUrl: 'http://localhost:2000',
      apiBaseUrl: 'https://79b1bea58ae5.ngrok-free.app',
    };
  },
  methods: {
    async generateImage() {
      if (!this.imagePrompt.trim()) {
        alert("이미지 생성 커맨드를 입력해주세요.");
        return;
      }

      this.$emit('update:error', null); // 부모의 에러 메시지 초기화
      this.isLoading = true;
      this.Account = null;
      this.generatedImageUrl = '';

      const payload = {
        Account: "TempPrompt",
        Prompt: this.imagePrompt,
      };

      try {
        const response = await axios.post(`${this.apiBaseUrl}/PromptAPI`, payload);
        console.log(response.data.generatedImageUrl)
        this.generatedImageUrl = 'data:image/jpeg;base64,'+response.data.generatedImageUrl;
      } catch (err) {
        console.error("이미지 생성 API 호출 오류:", err);
        this.$emit('update:error', "이미지 생성에 실패했습니다."); // 부모로 에러 전파
      } finally {
        this.isLoading = false;
      }
    }
  }
}
</script> -->


<style scoped>
/* CreationPanel에만 적용되는 스타일 */
.loader-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #888;
  font-size: 16px;
}

.loader {
  border: 6px solid #f0f2f5;
  border-top: 6px solid var(--primary-color);
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.command-input {
  width: 100%;
  box-sizing: border-box;
  padding: 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  margin-bottom: 16px;
  background-color: #fafafa;
}

.command-input input {
  width: 100%;
  border: none;
  outline: none;
  background: transparent;
  font-size: 16px;
}
</style>