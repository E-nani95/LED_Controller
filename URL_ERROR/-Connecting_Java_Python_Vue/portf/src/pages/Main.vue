<template>
  <div id="app">
    <div class="tab-navigation">
      <button 
        :class="{ active: activeTab === 'detection' }" 
        @click="activeTab = 'detection'">
        Deepfake Detection
      </button>
      <button 
        :class="{ active: activeTab === 'creation' }" 
        @click="activeTab = 'creation'">
        Create Image
      </button>
    </div>

    <div class="main-container">
      <DetectionPanel 
        v-if="activeTab === 'detection'" 
        @update:error="handleError"
      />
      <CreationPanel 
        v-if="activeTab === 'creation'" 
        @update:error="handleError"
      />
    </div>
    
    <div v-if="error" class="status-message error">
      <p>오류: {{ error }}</p>
    </div>
  </div>
</template>

<!-- <script>
// 자식 컴포넌트들을 import

import DetectionPanel from '@/components/DetectionPanel.vue';
import CreationPanel from '@/components/CreationPanel.vue';

export default {
  name: 'App',
  // 컴포넌트 등록
  components: {
    DetectionPanel,
    CreationPanel
  },
  data() {
    return {
      activeTab: 'detection', // 탭 상태 관리
      error: null             // 공통 에러 메시지 관리
    };
  },
  methods: {
    // 자식 컴포넌트에서 에러가 발생했을 때 호출될 메서드
    handleError(errorMessage) {
      this.error = errorMessage;
    }
  },
  watch: {
    // 탭이 변경될 때마다 에러 메시지를 초기화
    activeTab() {
      this.error = null;
    }
  }
}
</script> -->

<script setup>
import { ref, watch } from 'vue';

// 자식 컴포넌트들을 import합니다.
// <script setup> 내에서 import된 컴포넌트는 템플릿에서 바로 사용할 수 있습니다.
import DetectionPanel from '@/components/DetectionPanel.vue';
import CreationPanel from '@/components/CreationPanel.vue';

// data()에서 관리하던 상태들을 ref를 사용하여 반응형 변수로 선언합니다.
const activeTab = ref('detection'); // 탭 상태 관리
const error = ref(null);             // 공통 에러 메시지 관리

// methods에 있던 메서드를 일반 함수로 정의합니다.
// ref로 선언된 변수의 값에 접근하거나 변경할 때는 .value를 사용합니다.
const handleError = (errorMessage) => {
  error.value = errorMessage;
};

// watch 하던 속성을 watch() 함수를 사용하여 감시합니다.
// activeTab 값이 변경될 때마다 콜백 함수가 실행됩니다.
watch(activeTab, (newTab, oldTab) => {
  error.value = null; // 탭이 변경되면 에러 메시지를 초기화합니다.
});
</script>

<style>
/* 전역 및 공통 스타일 */
:root {
  --panel-bg: #ffffff;
  --placeholder-bg: #f0f2f5;
  --border-color: #d9d9d9;
  --text-color: #333;
  --primary-color: #1890ff;
  --primary-hover-color: #40a9ff;
  --secondary-color: #555;
  --secondary-hover-color: #777;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #f0f2f5;
}

#app {
  padding: 24px;
}

/* 탭 네비게이션 스타일 */
.tab-navigation {
  display: flex;
  justify-content: center;
  margin-bottom: 24px;
  border-bottom: 2px solid var(--border-color);
}

.tab-navigation button {
  padding: 12px 24px;
  border: none;
  background-color: transparent;
  font-size: 18px;
  font-weight: 500;
  cursor: pointer;
  color: var(--secondary-color);
  border-bottom: 4px solid transparent;
  transition: all 0.3s;
  margin-bottom: -2px;
}

.tab-navigation button:hover {
  color: var(--primary-color);
}

.tab-navigation button.active {
  color: var(--primary-color);
  border-bottom: 4px solid var(--primary-color);
}

/* 메인 레이아웃 */
.main-container {
  display: flex;
  justify-content: center;
}

/* 모든 패널에 적용되는 공통 스타일 */
.panel {
  flex-basis: 800px;
  background: var(--panel-bg);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  padding: 24px;
  display: flex;
  flex-direction: column;
}

.panel h2 {
  text-align: center;
  margin-top: 0;
  margin-bottom: 20px;
  color: var(--text-color);
}

/* 이미지 미리보기 박스 (공통) */
.image-preview-box {
  flex-grow: 1;
  width: 100%;
  min-height: 300px;
  border: 2px dashed var(--border-color);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--placeholder-bg);
  margin-bottom: 16px;
  overflow: hidden;
  position: relative;
}

.image-preview-box img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.placeholder {
  color: #888;
}

/* 버튼 및 액션 (공통) */
.actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

button {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  transition: all 0.3s;
  background-color: var(--primary-color);
  color: white;
}

button:hover {
  background-color: var(--primary-hover-color);
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

button.button-secondary {
  background-color: var(--secondary-color);
}
button.button-secondary:hover {
  background-color: var(--secondary-hover-color);
}

/* 에러 메시지 (공통) */
.status-message.error {
  text-align: center;
  color: #ff4d4f;
  margin-top: 20px;
}
</style>