import openai
import os
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

# .env 파일 로드 (선택적 - 없어도 환경 변수로 동작)
try:
    from dotenv import load_dotenv
    # 현재 파일의 부모 디렉토리에서 .env 파일 찾기
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # python-dotenv가 설치되지 않은 경우 무시
    pass

# OpenAI 클라이언트 설정 (v1.0+ 문법)
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class MedicalAIAssistant:
    def __init__(self):
        self.model = "gpt-4"
        self.max_tokens = 1000
        
        # 의료 관련 키워드 정의
        self.medical_keywords = [
            # 뇌 CT 관련
            'ct', '뇌', '두부', '머리', '영상', '스캔', '슬라이스',
            # 질병 관련
            '출혈', '경색', '뇌졸중', '혈관', '병변', '종양', '부종',
            '경막', '지주막', '뇌실', '뇌조직', '혈종',
            # 의료 용어
            'hu', '값', '밀도', '진단', '증상', '치료', '검사', '분석',
            '환자', '의료', '병원', '의사', '간호사', '방사선',
            # 해부학 용어
            '전두엽', '두정엽', '측두엽', '후두엽', '소뇌', '뇌간',
            '대뇌', '중뇌', '연수', '교뇌',
            # 기본 의료 행위
            '수술', '약물', '처방', '입원', '외래', '응급',
            # 일반적인 의료 질문
            '아프', '통증', '증상', '질병', '건강', '몸',
            # 앱 사용법
            '방법', '어떻게', '사용', '분석', '단계', '순서', '도움', '안내'
        ]
        
    def is_medical_related(self, user_message: str) -> bool:
        """의료 관련 질문인지 엄격하게 판단"""
        message_lower = user_message.lower()
        
        # 확실히 비의료 관련인 것들 차단
        non_medical_keywords = [
            '날씨', '음식', '요리', '영화', '드라마', '게임', '스포츠',
            '정치', '경제', '주식', '부동산', '여행', '쇼핑', '패션',
            '음악', '연예인', '축구', '야구', '농구', '코인', '비트코인',
            '카페', '맛집', '레스토랑', '커피', '술', '맥주', '와인'
        ]
        
        # 확실히 비의료인 경우 차단
        for keyword in non_medical_keywords:
            if keyword in message_lower:
                return False
        
        # 의료 관련 키워드가 있는지 확인
        medical_keywords = [
            # 뇌 CT 관련
            'ct', '뇌', '두부', '머리', '영상', '스캔', '슬라이스',
            # 질병 관련  
            '출혈', '경색', '뇌졸중', '혈관', '병변', '종양', '부종',
            '경막', '지주막', '뇌실', '뇌조직', '혈종', '실질',
            # 의료 용어
            'hu', '값', '밀도', '진단', '증상', '치료', '검사', '분석',
            '환자', '의료', '병원', '의사', '간호사', '방사선',
            # 해부학 용어
            '전두엽', '두정엽', '측두엽', '후두엽', '소뇌', '뇌간',
            '대뇌', '중뇌', '연수', '교뇌',
            # 앱 사용법
            '방법', '어떻게', '사용', '단계', '순서', '도움', '안내',
            # 간단한 인사
            '안녕', '고마워', '감사', '고맙'
        ]
        
        # 의료 키워드 포함 여부 확인
        for keyword in medical_keywords:
            if keyword in message_lower:
                return True
        
        # 환자 정보나 결과 관련 키워드
        if any(word in message_lower for word in ['환자', '정보', '결과', '소견']):
            return True
        
        # 짧은 인사말은 허용 (5자 이하)
        if len(user_message.strip()) <= 5:
            simple_greetings = ['네', '예', '응', '넹', '좋아', '알겠']
            if any(greeting in message_lower for greeting in simple_greetings):
                return True
        
        # 위 조건에 해당하지 않으면 비의료로 판단
        return False
    
    def classify_question_type(self, user_message: str) -> str:
        """질문 유형 분류 (우선순위 개선)"""
        message_lower = user_message.lower()
        
        # 1순위: 진단 결과 요청 (가장 구체적인 것부터)
        if any(word in message_lower for word in ['진단', '결과', '소견']) and '어떻게' not in message_lower:
            return "diagnosis_result"
        
        # 2순위: 환자 정보 요청
        if any(word in message_lower for word in ['환자', '나이', '성별', '정보']):
            return "patient_info"
        
        # 3순위: HU 값 관련
        if any(word in message_lower for word in ['hu', '값', '범위']):
            return "hu_values"
        
        # 4순위: 방법/절차 문의
        if any(word in message_lower for word in ['어떻게', '방법', '절차', '과정', '순서', '단계']):
            return "method"
        
        # 5순위: 용어 설명 요청 (가장 넓은 범위)
        if ('?' in user_message or '뭐' in message_lower or '무엇' in message_lower or 
            '설명' in message_lower or '의미' in message_lower or '정의' in message_lower):
            return "explanation"
        
        return "general"

    def get_non_medical_response(self) -> str:
        """의료 외 질문에 대한 거절 응답"""
        return """의료 AI 어시스턴트 안내

죄송하지만, 저는 뇌 CT 영상 분석 및 의료 교육만을 전문으로 하는 AI입니다.

답변 가능한 질문들:
• 뇌 CT 분석 방법
• HU 값 해석
• 뇌출혈/뇌경색 진단
• 환자 정보 확인
• 영상 분석 결과 설명
• 의료 영상학 교육 내용

예시 질문:
• "이 환자의 진단은 무엇인가요?"
• "HU 값은 어떻게 해석하나요?"
• "뇌출혈의 종류를 알려주세요"

의료 관련 질문을 해주시면 자세히 도워드리겠습니다."""

    def is_simple_query(self, user_message: str) -> bool:
        """정말 간단한 질문만 룰 기반으로 처리 (대부분 OpenAI API 사용)"""
        message_lower = user_message.lower()
        
        # 아주 간단한 인사만 룰 기반
        simple_greetings = ['안녕', '고마워', '감사', '고맙', '네', '예', '응']
        if len(user_message.strip()) <= 5 and any(greeting in message_lower for greeting in simple_greetings):
            return True
        
        # 환자 정보 요청만 룰 기반 (확실한 답변 필요)
        if any(word in message_lower for word in ['환자 정보', '환자정보']) and len(user_message) <= 15:
            return True
        
        # 나머지는 모두 OpenAI API로 처리하여 자연스러운 답변
        return False
    
    def get_rule_based_response(self, user_message: str, analysis_context: Dict = None) -> str:
        """간소화된 룰 기반 응답 (환자 정보와 분석 결과만 처리)"""
        message_lower = user_message.lower()
        
        # 간단한 인사 관련만
        if any(keyword in message_lower for keyword in ['안녕', '고마워', '감사', '고맙']):
            return "언제든지 도움이 필요하시면 말씀해주세요."
        
        # 환자 정보 요청 (확실한 정보 제공)
        if '환자 정보' in message_lower or '환자정보' in message_lower:
            if analysis_context and analysis_context.get('patient_number'):
                return f"""현재 환자 정보

환자 번호: {analysis_context['patient_number']}
나이: {analysis_context['age']}세  
성별: {analysis_context['gender']}
진단: {analysis_context['diagnosis']}
골절: {'있음' if analysis_context.get('fracture', False) else '없음'}

더 자세한 분석이 필요하시면 구체적으로 질문해주세요."""
            else:
                return "환자 정보를 불러오는 중입니다. 이미지를 먼저 선택해주세요."
        
        # 분석 결과 요청 (풍성한 교육 답변)
        if any(word in message_lower for word in ['결과', '분석', '진단', '소견']):
            if not analysis_context or not analysis_context.get('patient_number'):
                return "환자 정보를 먼저 불러와야 합니다. 이미지 드롭다운에서 환자를 선택해주세요."
            
            # 분석이 완료되지 않은 상태
            if not analysis_context.get('has_analysis', False):
                diagnosis = analysis_context.get('diagnosis', '정상')
                detailed = analysis_context.get('detailed_diagnosis', {})
                
                if diagnosis != '정상' and detailed:
                    detail_text = []
                    for hemorrhage_type, details in detailed.items():
                        detail_text.append(f"• {hemorrhage_type}: {details['affected_slices']}개 슬라이스 ({details['percentage']}%)")
                    
                    return f"""현재 환자의 의료 기록 정보

진단: {diagnosis}

상세 정보:
{chr(10).join(detail_text)}

※ 이는 기존 의료 기록 정보입니다. 
※ 정확한 분석을 위해 3단계 분석을 완료해주세요:
  1단계: 축방향 윤곽선 그리기
  2단계: 시상면 높이 설정  
  3단계: HU 값 범위 선택

※ 교육 목적 정보입니다. 실제 진단은 전문의와 상담하세요."""
                else:
                    return f"""현재 환자의 의료 기록 정보

진단: {diagnosis}

※ 더 정확한 분석을 위해 3단계 분석을 완료해보세요:
  1단계: 축방향 윤곽선 그리기
  2단계: 시상면 높이 설정  
  3단계: HU 값 범위 선택"""
            
            # 분석이 완료된 상태 - 풍성한 교육용 답변
            else:
                real_diagnosis = analysis_context.get('real_diagnosis', '정보 없음')
                ai_result = analysis_context.get('ai_analysis_result', '정보 없음')
                hu_range = analysis_context.get('actual_hu_range', {})
                lesion_volume = analysis_context.get('lesion_volume', 0)
                slice_range = analysis_context.get('slice_range', {})
                learning_point = analysis_context.get('learning_point', '추가 학습 정보 없음')
                csv_diagnosis = analysis_context.get('diagnosis', '정상')
                detailed = analysis_context.get('detailed_diagnosis', {})
                
                response = f"""🏥 완료된 영상 분석 결과 (교육용)

■ 최종 방사선학적 진단
{real_diagnosis}

■ AI 보조 분석 결과  
{ai_result}

■ 정량적 분석 데이터"""
                
                # HU 값 범위 정보
                if hu_range:
                    min_hu = hu_range.get('min', 0)
                    max_hu = hu_range.get('max', 0)
                    response += f"""
• 선택된 HU 값 범위: {min_hu:.1f} ~ {max_hu:.1f}
• HU 값 해석:"""
                    
                    if max_hu > 50:
                        response += "\n  - 급성 출혈 범위 포함 (50+ HU)"
                    if min_hu < 30 and max_hu > 30:
                        response += "\n  - 정상 뇌조직 범위 포함 (30-40 HU)"
                    if min_hu < 20:
                        response += "\n  - 만성 출혈/부종 범위 포함 (<30 HU)"
                
                # 병변 크기 정보
                if lesion_volume > 0:
                    volume_ml = lesion_volume / 1000
                    response += f"""
• 병변 부피: {lesion_volume:,.0f} mm³ ({volume_ml:.1f} ml)
• 부피 평가:"""
                    
                    if volume_ml > 30:
                        response += " 대용량 출혈 (수술적 치료 고려 필요)"
                    elif volume_ml > 10:
                        response += " 중등도 출혈 (집중 관찰 필요)"
                    else:
                        response += " 소량 출혈 (보존적 치료 가능)"
                
                # 분포 범위 정보
                if slice_range:
                    total_slices = slice_range.get('end', 0) - slice_range.get('start', 0) + 1
                    response += f"""
• 슬라이스 분포: {slice_range.get('start', 0)} ~ {slice_range.get('end', 0)}번 ({total_slices}개 슬라이스)
• 분포 범위: {"광범위 분포" if total_slices > 10 else "국소 분포"}"""
                
                # CSV 기반 상세 진단과 비교
                if detailed:
                    response += f"\n\n■ 기존 의료 기록과의 비교"
                    response += f"\n• 기록상 진단: {csv_diagnosis}"
                    for hemorrhage_type, details in detailed.items():
                        response += f"\n• {hemorrhage_type}: {details['affected_slices']}개 슬라이스 ({details['percentage']}%)"
                
                # 학습 포인트
                response += f"""

■ 학습 포인트 및 임상적 의의
{learning_point}

■ 추가 교육 정보
• CT에서 급성 출혈은 높은 HU 값(50-90)으로 밝게 나타납니다
• 시간이 지나면서 HU 값이 감소하여 만성 출혈로 변화합니다
• 병변의 위치와 크기는 치료 방향 결정에 중요합니다
• 실제 임상에서는 환자 증상과 함께 종합적으로 판단합니다

※ 이는 교육 목적 분석이며, 실제 진단은 반드시 전문의와 상담하세요."""
                
                return response
        
        # 나머지는 모두 OpenAI API로 넘김
        return None

    def generate_response(self, analysis_context: Dict, user_message: str, chat_history: List = None) -> str:
        """OpenAI API를 사용한 응답 생성"""
        try:
            # API 키 확인
            if not client.api_key:
                return "OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정해주세요."
            
            # 시스템 프롬프트 구성
            system_prompt = self._build_system_prompt(analysis_context)
            
            # 대화 히스토리 포함
            messages = [{"role": "system", "content": system_prompt}]
            
            if chat_history:
                for msg in chat_history[-10:]:  # 최근 10개만 포함
                    if msg.get('user'):
                        messages.append({"role": "user", "content": msg['user']})
                    if msg.get('assistant'):
                        messages.append({"role": "assistant", "content": msg['assistant']})
            
            messages.append({"role": "user", "content": user_message})
            
            # OpenAI API 호출 (v1.0+ 문법)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # 볼드 마크다운 제거
            ai_response = ai_response.replace("**", "")
            
            return ai_response
            
        except Exception as e:
            # OpenAI API v1.0+ 에러 처리
            error_msg = str(e)
            if "authentication" in error_msg.lower():
                return "API 키 인증에 실패했습니다. OpenAI API 키를 확인해주세요."
            elif "rate_limit" in error_msg.lower():
                return "API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            elif "quota" in error_msg.lower():
                return "API 사용량 한도를 초과했습니다. 계정 설정을 확인해주세요."
            else:
                print(f"OpenAI API 오류: {e}")
                # API 오류 시 룰 기반 응답으로 폴백
                return self.get_rule_based_response(user_message, analysis_context)

    def _build_system_prompt(self, analysis_context: Dict) -> str:
        """의료 AI 시스템 프롬프트 구성 (볼드 마크다운 제거)"""
        patient_info = ""
        analysis_info = ""
        
        if analysis_context:
            # 기본 환자 정보
            patient_info = f"""현재 환자 정보:
환자 번호: {analysis_context.get('patient_number', 'N/A')}
나이: {analysis_context.get('age', 'N/A')}세
성별: {analysis_context.get('gender', 'N/A')}
기본 진단: {analysis_context.get('diagnosis', '정상')}
골절: {'있음' if analysis_context.get('fracture', False) else '없음'}

상세 진단 정보:
{self._format_detailed_diagnosis(analysis_context.get('detailed_diagnosis', {}))}"""

            # 분석이 완료된 경우 실제 분석 결과 추가
            if analysis_context.get('has_analysis', False):
                analysis_info = "\n\n=== 완료된 영상 분석 결과 ==="
                
                if analysis_context.get('real_diagnosis'):
                    analysis_info += f"\n실제 방사선학적 진단: {analysis_context['real_diagnosis']}"
                
                if analysis_context.get('ai_analysis_result'):
                    analysis_info += f"\nAI 분석 결과: {analysis_context['ai_analysis_result']}"
                
                if analysis_context.get('actual_hu_range'):
                    hu_range = analysis_context['actual_hu_range']
                    analysis_info += f"\n선택된 HU 값 범위: {hu_range['min']:.1f} ~ {hu_range['max']:.1f}"
                
                if analysis_context.get('lesion_volume'):
                    volume = analysis_context['lesion_volume']
                    volume_ml = volume / 1000
                    analysis_info += f"\n병변 부피: {volume:,.0f} mm³ ({volume_ml:.1f} ml)"
                
                if analysis_context.get('slice_range'):
                    slice_info = analysis_context['slice_range']
                    total_slices = slice_info['end'] - slice_info['start'] + 1
                    analysis_info += f"\n슬라이스 분포: {slice_info['start']} ~ {slice_info['end']}번 ({total_slices}개)"
                
                if analysis_context.get('learning_point'):
                    analysis_info += f"\n학습 포인트: {analysis_context['learning_point']}"
                
                analysis_info += "\n\n이 실제 분석 결과를 활용하여 교육적이고 상세한 답변을 제공하세요."
        
        base_prompt = """당신은 뇌 CT 영상 분석 전문 의료 AI 어시스턴트입니다.

역할:
- 의료진 및 학습자를 위한 상세하고 교육적인 정보 제공
- 뇌 CT 영상 분석 결과의 정확한 해석과 설명
- 임상적 의의, 치료 방향, 예후에 대한 포괄적 안내
- 의료 용어와 개념에 대한 자연스럽고 이해하기 쉬운 설명

응답 특징:
- 자연스럽고 친근한 대화체로 응답
- 볼드체 마크다운(**) 사용하지 않기
- 분석 완료 시 정량적 데이터와 임상적 해석을 풍부하게 제공
- 환자 안전과 교육적 가치를 최우선으로 고려
- 실제 측정값의 의미와 중요성을 구체적으로 설명

교육적 응답 가이드라인:
1. 의료 용어 질문: 정의, 원인, 특징, 진단법, 치료법을 종합적으로 설명
2. 환자별 분석: CSV 기록과 실제 분석 결과를 비교하여 학습 포인트 제시
3. 분석 결과 해석: HU 값, 부피, 분포의 임상적 의미를 상세히 설명
4. 치료적 관점: 병변 특성에 따른 치료 접근법과 예후 정보 제공
5. 안전성 강조: 교육 목적임을 명시하고 전문의 상담 권고

특별 지침:
- 경막외출혈, 뇌실질내출혈 등 모든 의료 용어에 대해 자세히 설명
- 환자 상태와 분석 결과를 연결하여 통합적 관점 제시
- 실무에서 중요한 감별점과 주의사항 포함
- 한국 의료 환경에 맞는 실용적 정보 제공"""
        
        return f"""{base_prompt}

{patient_info}{analysis_info}

주의사항: 이는 교육용 시스템이며, 실제 임상 진단에는 전문의 판단이 필요합니다."""

    def _format_detailed_diagnosis(self, detailed_diagnosis: Dict) -> str:
        """상세 진단 정보를 포맷팅"""
        if not detailed_diagnosis:
            return "상세 진단 정보 없음"
        
        result = []
        for hemorrhage_type, details in detailed_diagnosis.items():
            result.append(f"• {hemorrhage_type}: {details['affected_slices']}개 슬라이스 ({details['percentage']}%)")
        
        return "\n".join(result) if result else "상세 진단 정보 없음"

    def get_medical_response(self, user_message: str, analysis_context: Dict = None, chat_history: List = None) -> str:
        """메인 의료 응답 생성 메서드"""
        # 1단계: 의료 관련 질문인지 확인
        if not self.is_medical_related(user_message):
            return self.get_non_medical_response()
        
        # 2단계: 간단한 질문만 룰 기반으로 처리
        if self.is_simple_query(user_message):
            rule_response = self.get_rule_based_response(user_message, analysis_context)
            if rule_response is not None:  # 룰 기반 응답이 있으면 사용
                return rule_response
        
        # 3단계: 나머지는 모두 OpenAI API로 자연스럽게 처리
        if client.api_key:
            return self.generate_response(analysis_context or {}, user_message, chat_history)
        else:
            # API 키가 없으면 기본 안내 메시지
            return "OpenAI API 키가 설정되지 않아 상세한 답변을 제공할 수 없습니다. 환경변수를 확인해주세요."

# 전역 인스턴스 생성
medical_ai = MedicalAIAssistant()

def get_ai_response(user_message: str, analysis_context: Dict = None, chat_history: List = None) -> str:
    """외부에서 호출하는 메인 함수"""
    return medical_ai.get_medical_response(user_message, analysis_context, chat_history) 