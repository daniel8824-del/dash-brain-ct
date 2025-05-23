# 🧠 뇌 CT 분석 도구 (Brain CT Analysis Tool)

뇌졸중 및 뇌출혈 탐지를 위한 인터랙티브 웹 기반 CT 분석 애플리케이션입니다.

## 📋 프로젝트 개요

이 프로젝트는 의료진과 의학 학습자들이 뇌 CT 이미지를 분석하고 출혈 영역을 시각화할 수 있는 도구를 제공합니다. Dash 프레임워크를 기반으로 구축되었으며, AI 의료 어시스턴트가 포함되어 있어 학습과 진단에 도움을 줍니다.

### 🎯 주요 기능

- **🔍 다축 CT 이미지 뷰어**: 축방향(Axial) 및 시상면(Sagittal) 뷰 제공
- **🎨 인터랙티브 영역 분할**: 관심 영역(ROI) 그리기 및 HU 값 기반 병변 분할
- **📊 실시간 히스토그램 분석**: HU(Hounsfield Unit) 값 분포 시각화
- **🎭 3D 병변 시각화**: 분할된 병변의 3차원 메쉬 모델 생성
- **🤖 AI 의료 어시스턴트**: 분석 결과에 대한 질의응답 및 학습 지원
- **📈 통계 분석**: 병변 부피, 슬라이스 범위 등 정량적 분석

## 🗂️ 폴더 구조

```
Brain_CT/
├── 📁 dash-brain-ct-data/          # CT 스캔 데이터 및 메타데이터
│   ├── 📁 ct_scans/                # CT 이미지 파일들 (NII 형식)
│   │   ├── 049.nii ~ 130.nii      # 환자별 CT 스캔 (49개 파일)
│   │   └── ...
│   ├── 📁 masks/                   # 마스크 파일들
│   ├── hemorrhage_diagnosis_raw_ct.csv     # 상세 출혈 진단 데이터 (2816 라인)
│   ├── Patient_demographics.csv    # 환자 인구통계 정보
│   ├── split_raw_data.py          # 데이터 분할 스크립트
│   ├── ct_ich.yml                 # 설정 파일
│   ├── Read_me.txt                # 데이터셋 설명서
│   ├── SHA256SUMS.txt             # 파일 무결성 체크섬
│   └── LICENSE.txt                # 데이터 라이센스
│
├── 📁 dash-brain-app/             # 웹 애플리케이션
│   ├── 📁 assets/                 # 정적 리소스
│   │   ├── sample_brain_ct.nii    # 기본 샘플 CT 이미지
│   │   ├── modal.md               # 도움말 모달 내용
│   │   ├── style.css              # 사용자 정의 스타일
│   │   ├── autoscale.js           # JavaScript 유틸리티
│   │   └── dash-logo-new.png      # 로고 이미지
│   ├── app.py                     # 메인 애플리케이션 (2241 라인)
│   ├── chatbot_ai.py              # AI 챗봇 모듈
│   ├── requirements.txt           # Python 의존성 패키지
│   ├── Procfile                   # Heroku 배포 설정
│   ├── Patient_demographics.csv   # 환자 데이터 (복사본)
│   └── README.md                  # 앱별 설명서
│
├── 📁 venv/                       # Python 가상환경
└── README.md                      # 이 파일
```

## 🚀 설치 및 실행

### 1. 시스템 요구사항

- Python 3.8 이상
- 최소 8GB RAM (대용량 CT 이미지 처리용)
- 웹 브라우저 (Chrome, Firefox, Edge 권장)

### 2. 설치 과정

```bash
# 1. 저장소 클론 또는 다운로드
cd Brain_CT

# 2. 가상환경 생성 및 활성화
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. 의존성 패키지 설치
pip install -r dash-brain-app/requirements.txt

# 4. 환경 변수 설정 (AI 어시스턴트 사용을 위해 필요)
# env_example.txt 파일을 .env로 복사하고 실제 API 키 입력
copy env_example.txt .env
# 또는 Linux/macOS: cp env_example.txt .env

# 5. .env 파일을 편집하여 실제 OpenAI API 키 입력
# OPENAI_API_KEY=sk-your_actual_api_key_here

# 6. 애플리케이션 실행
cd dash-brain-app
python app.py
```

### 🔑 OpenAI API 키 설정

AI 어시스턴트 기능을 사용하려면 OpenAI API 키가 필요합니다:

1. [OpenAI 웹사이트](https://platform.openai.com/account/api-keys)에서 계정 생성
2. API 키 발급
3. `.env` 파일에 키 입력:
   ```
   OPENAI_API_KEY=sk-your_actual_api_key_here
   ```

**⚠️ 중요**: `.env` 파일은 Git에 커밋하지 마세요! (이미 .gitignore에 포함됨)

### 3. 브라우저에서 접속

```
http://localhost:8050
```

## 🎮 사용법

### 📖 기본 워크플로우

1. **이미지 선택**: 드롭다운에서 분석할 환자의 CT 이미지 선택
2. **환자 정보 확인**: 나이, 성별, 기존 진단 정보 검토
3. **축방향 분석**: 출혈/병변 영역 주변에 윤곽선 그리기
4. **시상면 분석**: 병변의 상하 경계를 포함하는 사각형 그리기
5. **HU 값 선택**: 히스토그램에서 병변에 해당하는 강도 범위 선택
6. **3D 시각화**: 생성된 3차원 병변 모델 확인
7. **AI 상담**: 분석 결과에 대해 AI 어시스턴트와 질의응답

### 🔬 HU 값 가이드

| 조직/병변 유형 | HU 값 범위 | 설명 |
|-------------|-----------|------|
| 급성 출혈 | 50-90 HU | 신선한 혈종, 고밀도 |
| 만성 출혈 | 20-40 HU | 오래된 혈종 |
| 뇌경색 | 10-30 HU | 저밀도 병변 |
| 정상 뇌조직 | 30-40 HU | 회백질, 백질 |
| 뇌척수액 | 0-15 HU | 뇌실, 지주막하 공간 |

## 📊 데이터셋 정보

### 환자 데이터
- **총 환자 수**: 49명 (환자 49번 ~ 130번)
- **이미지 형식**: NII (NIfTI) 포맷
- **평균 파일 크기**: 18-20MB per 환자
- **총 데이터 크기**: 약 1GB

### 진단 분류
- 뇌실내출혈 (Intraventricular)
- 뇌실질내출혈 (Intraparenchymal)
- 지주막하출혈 (Subarachnoid)
- 경막외출혈 (Epidural)
- 경막하출혈 (Subdural)
- 골절 여부 (Fracture)

## 🤖 AI 어시스턴트 기능

내장된 AI 의료 어시스턴트는 다음과 같은 도움을 제공합니다:

- **분석 결과 해석**: HU 값과 병변 패턴 설명
- **의학적 질의응답**: CT 판독 관련 학습 지원
- **진단 비교**: AI 분석 vs 실제 방사선학적 진단
- **학습 가이드**: 뇌 CT 판독 팁과 주의사항

## 🔧 기술 스택

### 백엔드
- **Python 3.8+**: 메인 프로그래밍 언어
- **Dash**: 웹 애플리케이션 프레임워크
- **Plotly**: 인터랙티브 시각화
- **Nilearn**: 신경영상 처리
- **Scikit-image**: 이미지 처리 및 분할
- **Pandas**: 데이터 조작 및 분석
- **NumPy**: 수치 계산

### 프론트엔드
- **Dash Bootstrap Components**: UI 컴포넌트
- **HTML/CSS**: 사용자 인터페이스
- **JavaScript**: 클라이언트 사이드 기능

### AI 기능
- **OpenAI API**: GPT 기반 의료 어시스턴트
- **자연어 처리**: 의료 질의응답

## 📝 주요 파일 설명

### `dash-brain-app/app.py` (2241 라인)
- 메인 애플리케이션 로직
- 이미지 로딩 및 처리
- 인터랙티브 콜백 함수들
- 3D 시각화 생성

### `dash-brain-app/chatbot_ai.py`
- AI 챗봇 백엔드
- OpenAI API 연동
- 의료 컨텍스트 처리

### `dash-brain-ct-data/hemorrhage_diagnosis_raw_ct.csv`
- 슬라이스별 상세 진단 정보
- 각 출혈 타입별 라벨링 데이터
- 2816개 슬라이스 정보

## 🔒 보안 및 개인정보

⚠️ **중요 사항**:
- 이 도구는 **교육 및 연구 목적**으로만 사용되어야 합니다
- **실제 임상 진단에 사용하지 마세요**
- 환자 데이터는 익명화되어 제공됩니다
- AI 어시스턴트는 학습 지원용이며 의학적 조언을 제공하지 않습니다

🔐 **API 키 보안**:
- **절대로 API 키를 코드에 하드코딩하지 마세요**
- `.env` 파일은 Git에 커밋하지 마세요 (이미 .gitignore에 포함됨)
- API 키를 공유하거나 공개된 곳에 노출하지 마세요
- 의심스러운 활동이 있으면 즉시 API 키를 재발급하세요
- OpenAI 계정에서 사용량을 정기적으로 모니터링하세요

## 🆘 문제 해결

### 자주 발생하는 문제

1. **이미지 로딩 실패**
   ```bash
   # 경로 확인
   ls dash-brain-ct-data/ct_scans/
   ```

2. **가상환경 활성화 실패**
   ```bash
   # Windows에서 실행 정책 오류 시
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **메모리 부족**
   - 큰 CT 이미지 처리 시 8GB 이상 RAM 권장
   - 동시에 여러 환자 이미지 로딩 금지

4. **포트 충돌**
   ```bash
   # 다른 포트로 실행
   python app.py --port 8051
   ```

## 📄 라이센스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 상업적 사용은 금지됩니다.

---

**⚕️ 의료진 면책조항**: 이 소프트웨어는 교육 도구로만 사용되어야 하며, 실제 의료 진단이나 치료 결정에 사용해서는 안 됩니다. 