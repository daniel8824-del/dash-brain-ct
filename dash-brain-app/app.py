from time import time
import os
import numpy as np
import pandas as pd
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash_slicer import VolumeSlicer
from chatbot_ai import get_ai_response

# Bootstrap 스타일시트 설정
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets, 
                external_scripts=[
                    {'src': 'https://code.jquery.com/jquery-3.6.0.min.js'}
                ])

# app 서버 설정
server = app.server

# VolumeSlicer를 위한 Slicer 클래스 정의
class Slicer:
    pass

# 여기에 index_string 추가
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 슬라이더 스타일 조정 */
            .custom-slider {
                padding: 12px 0 20px 0 !important;
                margin-bottom: 15px !important;
                width: 100% !important;
            }
            .custom-slider .rc-slider-handle {
                border-color: #1890ff !important;
                background-color: #1890ff !important;
                width: 24px !important;
                height: 24px !important;
                margin-top: -8px !important;
            }
            .custom-slider .rc-slider-track {
                background-color: #1890ff !important;
                height: 8px !important;
            }
            .custom-slider .rc-slider-rail {
                height: 8px !important;
                background-color: #dee2e6 !important;
            }
            .graph-container {
                flex: 1;
                min-height: 300px;
                margin-bottom: 15px !important;
            }
            .slider-container {
                padding: 0 20px;
                margin: 10px 0 15px 0 !important;
                height: 35px !important;
                width: 100%;
            }
            /* AI 의료 어시스턴트 카드 스타일 */
            .ai-assistant-col {
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
                overflow: visible !important;
                display: flex !important;
                flex-direction: column !important;
            }
            .ai-assistant-card {
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
                overflow: visible !important;
                background-color: white !important;
            }
            .ai-assistant-card .card-header {
                background-color: transparent !important;
                border: none !important;
                padding: 12px 15px 8px 15px !important;
            }
            .ai-assistant-body {
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
                overflow: visible !important;
                padding: 15px !important;
                background-color: transparent !important;
            }
            .ai-chat-container {
                display: flex !important;
                flex-direction: column !important;
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
                overflow: visible !important;
                word-wrap: break-word !important;
            }
            /* 슬라이더 마크 강제 숨김 */
            .rc-slider-mark {
                display: none !important;
                visibility: hidden !important;
            }
            .rc-slider-mark-text {
                display: none !important;
            }
            /* RC 슬라이더 스타일 */
            .rc-slider {
                margin: 0 15px !important;
                height: 25px !important;
                padding: 8px 0 !important;
                width: 95% !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

t1 = time()

# ------------- 데이터셋 관리 ---------------------------------------------------

# 데이터셋 경로 설정
DATASET_DIR = "../dash-brain-ct-data"  # 올바른 상대 경로로 수정
DEFAULT_IMAGE = "assets/sample_brain_ct.nii"  # 기본 뇌 CT 이미지로 변경

# 사용 가능한 이미지 파일 목록 가져오기
def get_available_images():
    images = []
    
    if os.path.exists(DATASET_DIR):
        # CT 스캔 폴더에서 이미지 파일 검색
        ct_scans_dir = os.path.join(DATASET_DIR, "ct_scans")
        if os.path.exists(ct_scans_dir):
            image_files = [f for f in os.listdir(ct_scans_dir) 
                          if f.lower().endswith(('.nii', '.nii.gz'))]
            # 숫자 순으로 정렬
            image_files.sort(key=lambda x: int(x.split('.')[0]))
            
            # 메타데이터 로드 (환자 정보 포함)
            metadata_path = os.path.join(DATASET_DIR, "Patient_demographics.csv")
        if os.path.exists(metadata_path):
            try:
                    # CSV 파일의 헤더가 두 줄로 되어 있으므로 수동으로 컬럼명 지정
                    metadata = pd.read_csv(metadata_path, skiprows=1)  # 첫 번째 헤더 줄 건너뛰기
                    metadata.columns = [
                        'Patient_Number', 'Age', 'Gender', 'Intraventricular', 
                        'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 
                        'Fracture', 'Note1'
                    ]
                    # 각 파일에 대해 환자 정보와 함께 표시
                    for img_file in image_files:
                        patient_num = int(img_file.split('.')[0])
                        # 환자 정보 찾기
                        patient_info = metadata[metadata['Patient_Number'] == patient_num]
                        if not patient_info.empty:
                            row = patient_info.iloc[0]
                            age = int(float(row['Age'])) if pd.notna(row['Age']) and str(row['Age']).replace('.', '').replace('-', '').isdigit() else '알수없음'
                            gender = row['Gender'] if pd.notna(row['Gender']) else '알수없음'
                            # 출혈 타입 확인
                            hemorrhage_types = []
                            if pd.notna(row.get('Intraventricular')) and row.get('Intraventricular') == 1:
                                hemorrhage_types.append('뇌실내출혈')
                            if pd.notna(row.get('Intraparenchymal')) and row.get('Intraparenchymal') == 1:
                                hemorrhage_types.append('뇌실질내출혈')
                            if pd.notna(row.get('Subarachnoid')) and row.get('Subarachnoid') == 1:
                                hemorrhage_types.append('지주막하출혈')
                            if pd.notna(row.get('Epidural')) and row.get('Epidural') == 1:
                                hemorrhage_types.append('경막외출혈')
                            if pd.notna(row.get('Subdural')) and row.get('Subdural') == 1:
                                hemorrhage_types.append('경막하출혈')
                            
                            hemorrhage_str = ', '.join(hemorrhage_types) if hemorrhage_types else '정상'
                            label = f"환자 {patient_num} (나이 : {age}세, 성별 : {gender}, 진단 : {hemorrhage_str})"
                        else:
                            label = f"환자 {patient_num} (정보없음)"
                        
                        images.append({'label': label, 'value': img_file})
                    
                    return images
                    
            except Exception as e:
                print(f"메타데이터 로드 오류: {e}")
                    # 메타데이터 로드 실패 시 파일명만 사용
                return [{'label': f"환자 {f.split('.')[0]}", 'value': f} for f in image_files]
            else:
                # 메타데이터 파일이 없는 경우
                return [{'label': f"환자 {f.split('.')[0]}", 'value': f} for f in image_files]
    
    # 기본 이미지만 반환
    return [{'label': "기본 뇌 CT 샘플 이미지 (NII)", 'value': "기본 뇌 CT 샘플 이미지 (NII)"}]

available_images = get_available_images()

# 이미지 로드 함수
def load_image(image_name):
    try:
        if image_name == "기본 뇌 CT 샘플 이미지 (NII)":
            img = image.load_img(DEFAULT_IMAGE)
        else:
            img_path = os.path.join(DATASET_DIR, "ct_scans", image_name)
            print(f"🔄 이미지 로드 중: {img_path}")
            img = image.load_img(img_path)
        
        mat = img.affine
        img = img.get_fdata()
        img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]
        spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])
        print(f"이미지 크기: {img.shape}, 스페이싱: {spacing}")
        return img, spacing
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        # 기본 이미지로 폴백
        img = image.load_img(DEFAULT_IMAGE)
        mat = img.affine
        img = img.get_fdata()
        img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]
        spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])
        return img, spacing

# 환자 정보를 가져오는 함수 추가
def get_patient_info(image_name):
    """선택된 이미지 파일명으로부터 환자 정보를 가져옵니다."""
    if image_name == "기본 뇌 CT 샘플 이미지 (NII)":
        return {
            'patient_num': '샘플',
            'age': '알수없음',
            'gender': '알수없음',
            'hemorrhage_types': [],
            'fracture': False,
            'note': '기본 샘플 이미지입니다.',
            'detailed_diagnosis': {},
            'total_slices': 0,
            'affected_slices': 0
        }
    
    try:
        patient_num = int(image_name.split('.')[0])
        
        # 기본 환자 정보 로드
        demographics_path = os.path.join(DATASET_DIR, "Patient_demographics.csv")
        detailed_diagnosis_path = os.path.join(DATASET_DIR, "hemorrhage_diagnosis_raw_ct.csv")
        
        patient_data = {
            'patient_num': patient_num,
            'age': '알수없음',
            'gender': '알수없음',
            'hemorrhage_types': [],
            'fracture': False,
            'note': '',
            'detailed_diagnosis': {},
            'total_slices': 0,
            'affected_slices': 0
        }
        
        # 기본 환자 정보 로드
        if os.path.exists(demographics_path):
            # CSV 파일의 헤더가 두 줄로 되어 있으므로 수동으로 컬럼명 지정
            demographics = pd.read_csv(demographics_path, skiprows=1)  # 첫 번째 헤더 줄 건너뛰기
            demographics.columns = [
                'Patient_Number', 'Age', 'Gender', 'Intraventricular', 
                'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 
                'Fracture', 'Note1'
            ]
            patient_info = demographics[demographics['Patient_Number'] == patient_num]
            
            if not patient_info.empty:
                row = patient_info.iloc[0]
                patient_data['age'] = int(float(row['Age'])) if pd.notna(row['Age']) and str(row['Age']).replace('.', '').replace('-', '').isdigit() else '알수없음'
                patient_data['gender'] = row['Gender'] if pd.notna(row['Gender']) else '알수없음'
                patient_data['fracture'] = bool(pd.notna(row.get('Fracture')) and row.get('Fracture') == 1)
                patient_data['note'] = row.get('Note1') if pd.notna(row.get('Note1')) else ''
                
                # 출혈 타입 확인 (기본 정보) - 올바른 컬럼명 사용
                if pd.notna(row.get('Intraventricular')) and row.get('Intraventricular') == 1:
                    patient_data['hemorrhage_types'].append('뇌실내출혈')
                if pd.notna(row.get('Intraparenchymal')) and row.get('Intraparenchymal') == 1:
                    patient_data['hemorrhage_types'].append('뇌실질내출혈')
                if pd.notna(row.get('Subarachnoid')) and row.get('Subarachnoid') == 1:
                    patient_data['hemorrhage_types'].append('지주막하출혈')
                if pd.notna(row.get('Epidural')) and row.get('Epidural') == 1:
                    patient_data['hemorrhage_types'].append('경막외출혈')
                if pd.notna(row.get('Subdural')) and row.get('Subdural') == 1:
                    patient_data['hemorrhage_types'].append('경막하출혈')
        
        # 상세 진단 정보 로드 (슬라이스별)
        if os.path.exists(detailed_diagnosis_path):
            detailed_data = pd.read_csv(detailed_diagnosis_path)
            patient_slices = detailed_data[detailed_data['PatientNumber'] == patient_num]
            
            if not patient_slices.empty:
                patient_data['total_slices'] = len(patient_slices)
                
                # 각 출혈 타입별 영향받은 슬라이스 수 계산
                hemorrhage_columns = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
                hemorrhage_names = ['뇌실내출혈', '뇌실질내출혈', '지주막하출혈', '경막외출혈', '경막하출혈']
                
                for col, name in zip(hemorrhage_columns, hemorrhage_names):
                    affected_count = len(patient_slices[patient_slices[col] == 1])
                    if affected_count > 0:
                        patient_data['detailed_diagnosis'][name] = {
                            'affected_slices': affected_count,
                            'percentage': round((affected_count / patient_data['total_slices']) * 100, 1),
                            'slice_range': patient_slices[patient_slices[col] == 1]['SliceNumber'].tolist()
                        }
                
                # 전체 영향받은 슬라이스 수 (출혈이 있는 슬라이스)
                patient_data['affected_slices'] = len(patient_slices[patient_slices['No_Hemorrhage'] == 0])
                
                # 골절 정보 (슬라이스별)
                fracture_slices = len(patient_slices[patient_slices['Fracture_Yes_No'] == 1])
                if fracture_slices > 0:
                    patient_data['fracture_details'] = {
                        'affected_slices': fracture_slices,
                        'percentage': round((fracture_slices / patient_data['total_slices']) * 100, 1)
                    }
        
        return patient_data
            
    except Exception as e:
        print(f"환자 정보 로드 오류: {e}")
    
    return {
        'patient_num': '알수없음',
        'age': '알수없음',
        'gender': '알수없음',
        'hemorrhage_types': [],
        'fracture': False,
        'note': '',
        'detailed_diagnosis': {},
        'total_slices': 0,
        'affected_slices': 0
    }

def calculate_smart_axial_recommendation(detailed_diagnosis):
    """출혈 타입별 중요도와 영향 범위를 고려한 스마트한 축방향 추천 위치 계산"""
    try:
        if not detailed_diagnosis:
            return None
        
        # 출혈 타입별 임상적 중요도 가중치 (응급도 기준)
        hemorrhage_weights = {
            '경막외출혈': 1.0,      # 가장 응급 (생명 위험)
            '경막하출혈': 0.9,      # 매우 응급
            '뇌실질내출혈': 0.8,    # 응급
            '지주막하출혈': 0.7,    # 응급
            '뇌실내출혈': 0.6       # 상대적으로 덜 응급
        }
        
        weighted_center = 0
        total_weight = 0
        
        for hemorrhage_type, details in detailed_diagnosis.items():
            try:
                # 중요도 가중치
                importance_weight = hemorrhage_weights.get(hemorrhage_type, 0.5)
                
                # 영향 범위 가중치 (더 많은 슬라이스에 영향을 줄수록 높은 가중치)
                affected_slices = details.get('affected_slices', 1)
                range_weight = affected_slices / 10.0  # 정규화
                
                # 종합 가중치
                combined_weight = importance_weight * (1 + range_weight)
                
                # 해당 출혈의 중심 슬라이스
                slice_range = details.get('slice_range', [])
                if slice_range:
                    slice_center = (min(slice_range) + max(slice_range)) / 2
                    
                    weighted_center += slice_center * combined_weight
                    total_weight += combined_weight
            except Exception as e:
                print(f"출혈 타입 {hemorrhage_type} 처리 오류: {e}")
                continue
        
        if total_weight > 0:
            return int(weighted_center / total_weight)
        else:
            return None
            
    except Exception as e:
        print(f"축방향 추천 계산 오류: {e}")
        return None

def calculate_smart_sagittal_recommendation(detailed_diagnosis, patient_data, img_width):
    """출혈 위치와 환자 특성을 고려한 스마트한 시상면 추천 위치 계산"""
    try:
        if not detailed_diagnosis or not patient_data:
            return img_width // 2  # 기본 중앙값
        
        # 출혈 타입별 선호 시상면 위치 (해부학적 고려)
        sagittal_preferences = {
            '경막외출혈': 0.35,     # 측두엽 쪽에서 잘 보임
            '경막하출혈': 0.40,     # 약간 측면
            '뇌실질내출혈': 0.45,   # 중앙에서 약간 측면
            '지주막하출혈': 0.50,   # 중앙선 근처
            '뇌실내출혈': 0.55      # 중앙선에서 약간 안쪽
        }
        
        weighted_position = 0
        total_weight = 0
        
        for hemorrhage_type, details in detailed_diagnosis.items():
            # 출혈 타입별 선호 위치
            preferred_ratio = sagittal_preferences.get(hemorrhage_type, 0.45)
            
            # 영향 범위에 따른 가중치
            weight = details.get('affected_slices', 1)
            
            weighted_position += preferred_ratio * weight
            total_weight += weight
        
        if total_weight > 0:
            final_ratio = weighted_position / total_weight
            
            # 환자별 개별성 추가 (안전한 방식)
            try:
                patient_num = patient_data.get('patient_num', 0)
                if isinstance(patient_num, (int, str)):
                    patient_variation = (abs(hash(str(patient_num))) % 20 - 10) / 1000  # ±1% 변화
                    final_ratio += patient_variation
            except:
                pass  # 환자 변화 실패 시 무시
            
            # 범위 제한 (20% ~ 80%)
            final_ratio = max(0.2, min(0.8, final_ratio))
            
            return int(img_width * final_ratio)
        else:
            return img_width // 2
            
    except Exception as e:
        print(f"시상면 추천 계산 오류: {e}")
        return img_width // 2  # 오류 시 기본값 반환

# 기본 이미지 로드
img, spacing = load_image("기본 뇌 CT 샘플 이미지 (NII)")

# 이미지 처리
# Create smoothed image and histogram
med_img = filters.median(img, footprint=np.ones((1, 3, 3), dtype=bool))
hi = exposure.histogram(med_img)

# 슬라이스 중앙 위치 계산
axial_center = img.shape[0] // 2
sagittal_center = img.shape[1] // 2

# Create mesh
try:
    # 최신 버전 API 시도
    verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=5)
except Exception as e:
    print(f"첫 번째 marching_cubes 오류: {e}")
    # 이전 버전 API 시도
    try:
        verts, faces, _, _ = measure.marching_cubes_lewiner(volume=med_img, level=200, step_size=5)
    except Exception as e:
        print(f"두 번째 marching_cubes 오류: {e}")
        # 오류 발생 시 기본값 사용
        verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
x, y, z = verts.T
i, j, k = faces.T

# 3D 메쉬 초기화
fig_mesh = go.Figure()
fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

# 3D 뷰 레이아웃 개선
fig_mesh.update_layout(
    scene=dict(
        xaxis=dict(
            nticks=10, range=[0, img.shape[2]], 
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(150, 150, 150)",
            showbackground=True,
        ),
        yaxis=dict(
            nticks=10, range=[0, img.shape[1]], 
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(150, 150, 150)",
            showbackground=True,
        ),
        zaxis=dict(
            nticks=10, range=[0, img.shape[0]], 
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(150, 150, 150)",
            showbackground=True,
        ),
        aspectratio=dict(x=1, y=1, z=0.8),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            center=dict(x=0.5, y=0.5, z=0.5),
            up=dict(x=0, y=0, z=1)
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    plot_bgcolor="rgb(255, 255, 255)",
    paper_bgcolor="rgb(255, 255, 255)",
    autosize=True,
)

# 전역 슬라이서 변수 선언
slicer1 = None
slicer2 = None

# 슬라이서 생성 함수
def create_slicers(app, volume, spacing):
    """새로운 슬라이서를 생성하는 함수"""
    global slicer1, slicer2
    
    # 축방향 슬라이스 중앙 위치 계산
    axial_center = volume.shape[0] // 2
    sagittal_center = volume.shape[1] // 2
    
    # 축방향 슬라이서 생성
    slicer1 = VolumeSlicer(app, volume, axis=0, spacing=spacing, thumbnail=False)
    slicer1.graph.figure.update_layout(
        dragmode="drawclosedpath", 
        newshape_line_color="cyan", 
        plot_bgcolor="rgb(0, 0, 0)"
    )
    slicer1.graph.config.update(
        modeBarButtonsToAdd=["drawclosedpath", "eraseshape", "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        displayModeBar=True,
        displaylogo=False
    )
    slicer1.slider.marks = {}
    slicer1.slider.className = "custom-slider"
    slicer1.slider.value = axial_center
    
    # 시상면 슬라이서 생성
    slicer2 = VolumeSlicer(app, volume, axis=1, spacing=spacing, thumbnail=False)
    slicer2.graph.figure.update_layout(
        dragmode="drawrect", 
        newshape_line_color="cyan", 
        plot_bgcolor="rgb(0, 0, 0)"
    )
    slicer2.graph.config.update(
        modeBarButtonsToAdd=["drawrect", "eraseshape", "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        displayModeBar=True,
        displaylogo=False
    )
    slicer2.slider.marks = {}
    slicer2.slider.className = "custom-slider"
    slicer2.slider.value = sagittal_center
    
    return slicer1, slicer2

# 초기 슬라이서 생성
slicer1, slicer2 = create_slicers(app, img, spacing)

def path_to_coords(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)


def largest_connected_component(mask):
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (np.argmax(sizes) + 1)


t2 = time()
print("initial calculations", t2 - t1)

# ------------- 앱 레이아웃 정의 ---------------------------------------------------

# 이미지 선택 드롭다운 - 카드 대신 단순 드롭다운으로 변경
image_selection = html.Div([
    html.H6("뇌 CT 이미지 선택", className="mt-2 mb-2", style={"font-weight": "500", "font-size": "1.1rem"}),
                dcc.Dropdown(
                    id='image-dropdown',
        options=available_images,  # 이미 올바른 형식 (label, value)
        value=available_images[0]['value'] if available_images else None,
        placeholder="분석할 이미지를 선택하세요",
        className="mb-4"
    ),
    # 환자 정보 카드 추가
    dbc.Card([
        dbc.CardHeader([
            html.H6("환자 정보", className="mb-0", style={"color": "#495057"})
        ], className="bg-light", style={
            "padding": "0.25rem 1rem", 
            "margin": "0", 
            "height": "auto",
            "minHeight": "auto"
        }),
        dbc.CardBody([
            html.Div(id="patient-info", style={"fontSize": "0.9rem"})
        ], style={
            "padding": "0.25rem 1rem", 
            "margin": "0", 
            "height": "auto",
            "minHeight": "auto",
            "paddingBottom": "0.25rem"
        })
    ], className="shadow-sm", style={
        "border": "1px solid #dee2e6", 
        "marginBottom": "0.5rem",
        "height": "auto",
        "minHeight": "auto",
        "maxHeight": "none"
    })
], style={"padding": "0 10px"})  # 원래 패딩 복원

# 축방향 뷰 카드 (설명 상단 배치)
axial_card = dbc.Card(
    [
        dbc.CardHeader([
            html.H5("뇌의 축방향 뷰", className="mb-2", style={"marginTop": "10px"}),
            dbc.Alert([
                html.Strong("1단계: "), 
                "모든 축방향 슬라이스에서 ",
                html.Strong("뇌졸중 또는 출혈 영역"),
                "을 포함하는 윤곽선을 그립니다."
            ], color="info", className="py-2 mb-0")
        ], className="bg-light"),
        dbc.CardBody([
            # 도움말 아이콘
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-question-circle"), 
                    id="axial-help", 
                    color="link", 
                    size="sm",
                    className="float-right position-absolute top-0 end-0 m-2"
                ),
                dbc.Tooltip(
                    "슬라이더를 사용하여 영상을 세로로 스크롤하며 뇌졸중/출혈 영역을 찾으세요. 이후 해당 영역 주변에 닫힌 윤곽선을 그립니다.",
                    target="axial-help",
                ),
                # 슬라이서 그래프
                html.Div([
                    slicer1.graph,
                ], className="graph-container"),
                # 슬라이더를 그래프 바로 아래에 위치
                html.Div([
                    slicer1.slider,
                ], className="slider-container"),
            ], className="position-relative h-100 d-flex flex-column"),
            *slicer1.stores
        ], style={"height": "60vh", "min-height": "450px", "padding": "0"}),
    ],
    className="h-100 shadow-sm",
    id="axial-card"
)

# 시상면 뷰 카드 (설명 상단 배치)
saggital_card = dbc.Card(
    [
        dbc.CardHeader([
            html.H5("뇌의 시상면 뷰", className="mb-2", style={"marginTop": "10px"}),
            dbc.Alert([
                html.Strong("2단계: "), 
                "사각형을 그려서 병변 영역의 ",
                html.Strong("최소 및 최대 높이"),
                "를 지정합니다."
            ], color="info", className="py-2 mb-0")
        ], className="bg-light"),
        dbc.CardBody([
            # 도움말 아이콘
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-question-circle"), 
                    id="sagittal-help", 
                    color="link", 
                    size="sm",
                    className="float-right position-absolute top-0 end-0 m-2"
                ),
                dbc.Tooltip(
                    "사각형의 너비는 무시되고 높이만 사용됩니다. 병변 영역의 상하 경계를 포함하도록 사각형을 그려주세요.",
                    target="sagittal-help",
                ),
                # 슬라이서 그래프
                html.Div([
                    slicer2.graph,
                ], className="graph-container"),
                # 슬라이더를 그래프 바로 아래에 위치
                html.Div([
                    slicer2.slider,
                ], className="slider-container"),
            ], className="position-relative h-100 d-flex flex-column"),
            *slicer2.stores
        ], style={"height": "60vh", "min-height": "450px", "padding": "0"}),
    ],
    className="h-100 shadow-sm",
    id="saggital-card"
)

# 히스토그램 카드 (설명 상단 배치)
histogram_card = dbc.Card(
    [
        dbc.CardHeader([
            html.H5("강도 값 히스토그램", className="mb-2", style={"marginTop": "10px"}),
            dbc.Alert([
                html.Strong("3단계: "), 
                "병변 영역을 분할하기 위한 ",
                html.Strong("HU 값 범위"),
                "를 선택합니다."
            ], color="info", className="py-2 mb-0"),
            # 경고 메시지를 상단에 배치
            dbc.Collapse(
                dbc.Alert(
                    "먼저 1단계와 2단계에서 관심 영역을 정의해야 합니다!",
                    id="roi-warning",
                    color="danger", 
                    is_open=True,
                    className="mt-2 mb-0 py-2"
                ),
                id="warning-collapse",
                is_open=True,
            ),
        ], className="bg-light"),
        dbc.CardBody([
            dcc.Graph(
                id="graph-histogram",
                figure=px.bar(
                    x=hi[1],
                    y=hi[0],
                    labels={"x": "HU 값", "y": "빈도"},
                    template="plotly_white",
                ),
                config={
                    "modeBarButtonsToAdd": [
                        "drawline",
                        "drawclosedpath",
                        "drawrect",
                        "eraseshape",
                    ]
                },
                style={"height": "40vh", "min-height": "300px"}
            ),
        ]),
    ],
    className="h-100 shadow-sm",
    id="histogram-card"
)

# 3D 메쉬 카드
mesh_card = dbc.Card(
    [
        dbc.CardHeader([
            html.H5("3D 병변 시각화", className="mb-2", style={"marginTop": "10px"}),
            dbc.Alert([
                "분석 결과를 3차원으로 시각화합니다. ",
                html.Strong("마우스로 드래그하여 회전"),
                "할 수 있습니다."
            ], color="success", className="py-2 mb-0")
        ], className="bg-light"),
        dbc.CardBody([
            # 도움말 아이콘 제거
            html.Div([
            dcc.Graph(
                id="graph-helper", 
                figure=fig_mesh,
                    style={"height": "40vh", "min-height": "300px"}
                )
            ], className="position-relative h-100")
        ], style={
            "height": "40vh", 
            "min-height": "300px", 
            "padding": "0", 
            "backgroundColor": "rgb(255, 255, 255)",
            "borderRadius": "0 0 0.5rem 0.5rem"
        }),
    ],
    className="h-100 shadow-sm",
    id="mesh-card"
)

# 분석 결과 카드
analysis_card = dbc.Card(
    [
        dbc.CardHeader([
            html.H5("분석 결과", className="mb-0")
        ], className="bg-primary text-white"),
        dbc.CardBody(
            [
                dbc.Row([
                    dbc.Col([
                        html.H6("영상 분석 결과", className="border-bottom pb-2", style={"fontSize": "1rem", "fontWeight": "500", "paddingLeft": "15px"}),
                        html.Div(id="analysis-results", style={"padding": "10px 15px"}),
                    ], lg=6),
                    dbc.Col([
                        html.H6("병변 통계", className="border-bottom pb-2", style={"fontSize": "1rem", "fontWeight": "500", "paddingLeft": "15px"}),
                        html.Div(id="infection-stats", style={"padding": "10px 15px"}),
                    ], lg=6),
                ]),
            ]
        ),
    ],
    className="mt-4 shadow-sm"
)

# 모달 정의
with open("assets/modal.md", "r", encoding="utf-8") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("닫기", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

# 앱 레이아웃 수정
app.layout = html.Div(
    [
        # FontAwesome CDN 추가 (아이콘용)
        html.Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
        ),
        
        # 네비게이션 바
        dbc.Navbar(
            dbc.Container(
                [
                    # 로고와 앱 제목
                    dbc.Row(
                        [
                            dbc.Col(
                                html.A(
                                    html.Img(
                                        src=app.get_asset_url("dash-logo-new.png"),
                                        height="30px",
                                    ),
                                    href="https://plotly.com/dash/",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                html.H3("뇌 CT 분석 도구", className="mb-0 text-white"),
                                className="ml-2",
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    # 우측 설명 문구와 도움말 버튼
                    dbc.Row(
                        [
                        dbc.Col(
                            html.P("뇌졸중 및 뇌출혈 탐색 및 분석", className="mb-0 text-light"),
                            className="ml-auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="fas fa-question-circle mr-1"), " 도움말"],
                                    id="howto-open",
                                    color="light",
                                    outline=True,
                                    size="sm",
                                    className="ml-2"
                                ),
                                width="auto",
                            ),
                        ],
                        className="ml-auto align-items-center"
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
        ),
        
        # 메인 레이아웃: 좌측(CT 분석), 우측(챗봇)
        dbc.Container([
            dbc.Row([
                # 좌측: 뇌 CT 분석 도구
                dbc.Col([
                    # 이미지 선택 섹션
                    dbc.Row([
                        dbc.Col(image_selection, width=12),
                    ], className="g-0"),  # gutter 제거
                    
                    # 2x2 그리드 레이아웃
                    html.Div([
                        dbc.Row([
                            # 첫 번째 행: 축방향 뷰와 시상면 뷰
                            dbc.Col([axial_card], lg=6, md=12, sm=12, className="mb-4 pe-1"),
                            dbc.Col([saggital_card], lg=6, md=12, sm=12, className="mb-4 ps-1"),
                        ], className="mb-3 g-0"),  # gutter 제거
                        
                        dbc.Row([
                            # 두 번째 행: 히스토그램과 3D 메쉬
                            dbc.Col([histogram_card], lg=6, md=12, sm=12, className="mb-4 pe-1"),
                            dbc.Col([mesh_card], lg=6, md=12, sm=12, className="mb-4 ps-1"),
                        ], className="g-0"),  # gutter 제거
                    ], style={"padding": "0 10px"}),  # 환자정보와 동일한 패딩 추가
                    
                    # 분석 결과
                    html.Div([
                        dbc.Row([dbc.Col(analysis_card, width=12)], className="g-0"),  # gutter 제거
                    ], style={"padding": "0 10px"}),  # 환자정보와 동일한 패딩 추가
                ], 
                width=8,  # 좌측 67% 할당
                className="pe-2"  # 우측 패딩 축소
                ),
                
                # 우측: 챗봇 영역
                dbc.Col([
                    # 상단 여백 (이미지 선택 영역과 높이 맞춤)
                    html.Div(style={"height": "19.5px"}),
                    
                    # AI 의료 어시스턴트 카드
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("AI 의료 어시스턴트", className="mb-0", style={"color": "#495057", "margin": "0", "padding": "0"})
                        ], style={"backgroundColor": "transparent", "border": "none", "padding": "12px 15px 8px 15px"}),
                        dbc.CardBody([
                            html.Div([
                                html.Div(
                                    id="chat-messages",
                                    children=[
                                        html.Div([
                                            html.Div([
                                                html.Div([
                                                    html.Span("🤖", style={"fontSize": "16px", "marginRight": "6px"}),
                                                    html.Span([
                                                        "안녕하세요! 저는 뇌 CT 학습을 도와주는 어시스턴트입니다.",
                                                        html.Br(),
                                                        "CT 분석 결과에 대해 궁금한 점이 있으시면 언제든 질문해주세요!"
                                                    ])
                                                ], style={
                                                    "backgroundColor": "#f1f1f1", 
                                                    "color": "#333",
                                                    "padding": "8px 12px",
                                                    "borderRadius": "18px 18px 18px 4px",
                                                    "display": "inline-block",
                                                    "maxWidth": "85%",
                                                    "wordWrap": "break-word",
                                                    "fontSize": "14px",
                                                    "lineHeight": "1.4"
                                                })
                                            ], style={
                                                "textAlign": "left",
                                                "marginBottom": "8px"
                                            })
                                        ])
                                    ],
                                    style={
                                        "padding": "12px",
                                        "backgroundColor": "transparent",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "8px",
                                        "flex": "1"
                                    }
                                ),
                                html.Div([
                                    html.Div([
                                        dbc.Input(
                                            id="chat-input",
                                            placeholder="메시지 입력...",
                                            value="",
                                            style={
                                                "borderRadius": "20px",
                                                "border": "1px solid #e1e5e9",
                                                "fontSize": "14px",
                                                "backgroundColor": "#ffffff",
                                                "outline": "none",
                                                "boxShadow": "none",
                                                "paddingRight": "45px",
                                                "paddingLeft": "15px",
                                                "paddingTop": "8px",
                                                "paddingBottom": "8px"
                                            }
                                        ),
                                        dbc.Button(
                                            html.I(className="fas fa-paper-plane"),
                                            id="chat-send-btn",
                                            color="primary",
                                            style={
                                                "borderRadius": "50%",
                                                "border": "none",
                                                "width": "32px",
                                                "height": "32px",
                                                "padding": "0",
                                                "fontSize": "12px",
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "position": "absolute",
                                                "right": "6px",
                                                "top": "50%",
                                                "transform": "translateY(-50%)"
                                            }
                                        )
                                    ], style={
                                        "position": "relative",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "width": "100%"
                                    })
                                ], style={
                                    "backgroundColor": "transparent",
                                    "borderTop": "1px solid #e1e5e9",
                                    "padding": "8px 12px",
                                    "margin": "0"
                                })
                            ], className="ai-chat-container", style={
                                "display": "flex",
                                "flexDirection": "column",
                                "border": "1px solid #e1e5e9",
                                "borderRadius": "12px",
                                "backgroundColor": "transparent"
                            })
                        ], className="ai-assistant-body")
                    ], className="shadow-sm ai-assistant-card")
                ], width=4, className="ps-2 ai-assistant-col")
            ], className="g-0")  # Row gap 제거
        ], 
        fluid=True,
        className="pt-2 pb-4"
        ),
        
        # 저장소 및 모달
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
        dcc.Store(id="slice-state", data={"axial": axial_center, "sagittal": sagittal_center}),
        dcc.Store(id="chat-history", data=[]),
        dcc.Store(id="analysis-context", data={}),
        
        # 모달 다이얼로그
        modal_overlay,
    ],
    className="dash-brain-app",
    style={
        "fontFamily": "'Segoe UI', 'Roboto', sans-serif",
    }
)

t3 = time()
print("layout definition", t3 - t2)

# 모달 콜백 조정 (버튼 ID 변경 반영)
@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# ROI 경고 표시 여부 조건 콜백
@app.callback(
    Output("warning-collapse", "is_open"),
    [Input("annotations", "data")],
)
def toggle_roi_warning(annotations):
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        return True
    return False


# ------------- 앱 상호작용 정의 ---------------------------------------------------

# 이미지 선택 콜백 - 기본 정보 업데이트 (중복 없는 Output들)
@app.callback(
    [Output(slicer1.slider.id, "max"),
     Output(slicer2.slider.id, "max"),
     Output("patient-info", "children")],
    [Input("image-dropdown", "value")],
    prevent_initial_call=False  # 초기 로딩을 위해 False로 설정
)
def update_image_basic_info(selected_image):
    # 초기 로딩 시 기본 이미지 사용
    if selected_image is None:
        selected_image = available_images[0]['value'] if available_images else "기본 뇌 CT 샘플 이미지 (NII)"
    
    # 이미지 로드
    global img, spacing, med_img
    img, spacing = load_image(selected_image)
    med_img = filters.median(img, footprint=np.ones((1, 3, 3), dtype=bool))
    
    print("\n" + "=" * 60)
    print(f"🔄 이미지 변경: {selected_image}")
    print("=" * 60)
    print(f"📏 새 이미지 크기: {img.shape}")
    print(f"📐 새 이미지 스페이싱: {spacing}")
    print(f"🏥 HU 값 범위: {img.min():.1f} ~ {img.max():.1f}")
    print(f"📊 평균 HU 값: {img.mean():.1f}")
    
    # 슬라이서 설정 업데이트
    if slicer1 and slicer2:
        print(f"🔄 슬라이서 업데이트 중...")
        slicer1.volume = img
        slicer1.spacing = spacing
        slicer1._volume = img  # 내부 volume 참조도 업데이트
        slicer2.volume = img
        slicer2.spacing = spacing
        slicer2._volume = img  # 내부 volume 참조도 업데이트
        
        print(f"✅ 슬라이서 업데이트 완료")
    
    print(f"📍 Axial 중앙: {img.shape[0] // 2}, Sagittal 중앙: {img.shape[1] // 2}")
    
    # 환자 정보 가져오기 (로그 출력 전에 먼저 실행)
    patient_data = get_patient_info(selected_image)
    
    # 병변 중심 위치 계산 (시상면용)
    sagittal_center = calculate_smart_sagittal_recommendation(patient_data['detailed_diagnosis'], patient_data, img.shape[1])
    
    # 병변 최적 위치 로그 추가
    if patient_data['hemorrhage_types']:
        # 축방향 스마트 추천 계산
        axial_recommendation = calculate_smart_axial_recommendation(patient_data['detailed_diagnosis'])
        if axial_recommendation:
            print(f"🎯 병변 관찰 최적 위치 - Axial: {axial_recommendation} (스마트 분석)")
        print(f"🎯 병변 관찰 최적 위치 - Sagittal: {sagittal_center} (스마트 분석)")
    
    print("=" * 60 + "\n")
    
    # 최종 추천 위치 업데이트
    axial_center = img.shape[0] // 2
    
    # 진단 정보 구성
    if patient_data['hemorrhage_types']:
        diagnosis = ', '.join(patient_data['hemorrhage_types'])
        diagnosis_color = "danger"
        diagnosis_icon = "fa-exclamation-triangle"
    else:
        diagnosis = "정상 (출혈 소견 없음)"
        diagnosis_color = "success"
        diagnosis_icon = "fa-check-circle"
    
    # 상세 진단 정보 카드들 생성
    detailed_cards = []
    if patient_data['detailed_diagnosis']:
        for hemorrhage_type, details in patient_data['detailed_diagnosis'].items():
            slice_range_text = f"{min(details['slice_range'])}-{max(details['slice_range'])}" if len(details['slice_range']) > 1 else str(details['slice_range'][0])
            detailed_cards.append(
                dbc.Alert([
                    html.Div([
                        html.Strong(f"{hemorrhage_type}: "),
                        html.Span(f"{details['affected_slices']}개 슬라이스 ({details['percentage']}%)"),
                        html.Br(),
                        html.Small(f"슬라이스 범위: {slice_range_text}", className="text-muted")
                    ])
                ], color="warning", className="py-2 mb-2")
            )
    
    patient_info = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className=f"fas fa-user", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("환자 번호 : "),
                    html.Span(str(patient_data['patient_num']))
                ], className="mb-3", style={"paddingLeft": "5px"}),
                html.Div([
                    html.I(className=f"fas fa-birthday-cake", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("나이 : "),
                    html.Span(f"{patient_data['age']}세" if isinstance(patient_data['age'], int) else patient_data['age'])
                ], className="mb-3", style={"paddingLeft": "5px"}),
                html.Div([
                    html.I(className=f"fas fa-venus-mars", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("성별 : "),
                    html.Span(patient_data['gender'])
                ], className="mb-3", style={"paddingLeft": "5px"}),
            ], lg=6),
            dbc.Col([
                # 슬라이스 정보
                html.Div([
                    html.I(className=f"fas fa-layer-group", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("슬라이스 정보 : "),
                    html.Span(f"전체 {patient_data['total_slices']}개, 병변 {patient_data['affected_slices']}개")
                ], className="mb-3", style={"paddingLeft": "5px"}) if patient_data['total_slices'] > 0 else None,
                
                # 골절 정보
                html.Div([
                    html.I(className=f"fas fa-bone", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("골절 : "),
                    html.Span(
                        f"있음 ({patient_data.get('fracture_details', {}).get('affected_slices', 0)}개 슬라이스, "
                        f"{patient_data.get('fracture_details', {}).get('percentage', 0)}%)" 
                        if patient_data['fracture'] and 'fracture_details' in patient_data 
                        else ("있음" if patient_data['fracture'] else "없음"),
                        style={"color": "red" if patient_data['fracture'] else "green"}
                    )
                ], className="mb-3", style={"paddingLeft": "5px"}) if patient_data['patient_num'] != '샘플' else None,
                
                # 방사선학적 진단
                html.Div([
                    html.I(className=f"fas {diagnosis_icon}", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("방사선학적 진단 : "),
                    html.Span(diagnosis, style={
                        "color": "red" if diagnosis_color == "danger" else "green" if diagnosis_color == "success" else "#856404"
                    })
                ], className="mb-3", style={"paddingLeft": "5px"}),
                
                # 특이사항
                html.Div([
                    html.I(className=f"fas fa-clipboard", style={"color": "#6c757d", "width": "16px", "textAlign": "center", "marginRight": "8px"}),
                    html.Strong("특이사항 : "),
                    html.Span(patient_data['note'] if patient_data['note'] else "없음")
                ], className="mb-3", style={"paddingLeft": "5px"}) if patient_data['note'] else None,
            ], lg=6),
        ]),
        
        # 상세 진단 정보 (있는 경우)
        html.Div([
            html.Hr(className="my-2"),
            html.H6("상세 진단 정보", className="mb-2", style={"color": "#495057", "paddingLeft": "5px", "fontSize": "0.95rem"}),
            html.Div(detailed_cards, style={"paddingLeft": "5px", "paddingRight": "5px"}),
            
            # 추천 슬라이스 정보 추가
            html.Div([
                html.H6("분석 추천 위치", className="mb-2 mt-3", style={"color": "#495057", "paddingLeft": "5px", "fontSize": "0.95rem"}),
                
                # 축방향 추천 위치 계산 및 표시
                html.Div([
                    # 스마트한 병변 중심 축방향 위치 계산
                    dbc.Alert([
                        html.Strong("축방향 추천 위치: "),
                        html.Span(f"{calculate_smart_axial_recommendation(patient_data['detailed_diagnosis']) or axial_center}"),
                        html.Small(" (스마트 분석)" if patient_data['detailed_diagnosis'] else " (중앙)", 
                                className="text-muted" if patient_data['detailed_diagnosis'] else None)
                    ], color="primary" if patient_data['detailed_diagnosis'] else "info", 
                       className="py-2 mb-2", style={"fontSize": "0.9rem", "marginLeft": "5px", "marginRight": "5px"}),
                    
                    # 시상면 추천 위치 (색상을 보라색으로 통일)
                    dbc.Alert([
                        html.Strong("시상면 최적 위치: "),
                        html.Span(f"{calculate_smart_sagittal_recommendation(patient_data.get('detailed_diagnosis', {}), patient_data, img.shape[1]) if patient_data else img.shape[1] // 2}"),
                        html.Small(" (스마트 분석)" if patient_data.get('hemorrhage_types') else " (중앙)", 
                                className="text-muted" if patient_data.get('hemorrhage_types') else None)
                    ], color="primary" if patient_data.get('hemorrhage_types') else "info", 
                       className="py-2 mb-0", style={"fontSize": "0.9rem", "marginLeft": "5px", "marginRight": "5px"})
                ])
            ])
        ], className="mb-0") if detailed_cards or patient_data['patient_num'] != '샘플' else None
    ])
    
    return img.shape[0]-1, img.shape[1]-1, patient_info

# 이미지 선택 콜백 - 그래프와 슬라이더 업데이트 + shapes 초기화
@app.callback(
    [Output(slicer1.graph.id, "figure", allow_duplicate=True),
     Output(slicer2.graph.id, "figure", allow_duplicate=True),
     Output(slicer1.slider.id, "value", allow_duplicate=True),
     Output(slicer2.slider.id, "value", allow_duplicate=True)],
    [Input("image-dropdown", "value")],
    [State(slicer1.graph.id, "figure"),
     State(slicer2.graph.id, "figure")],
    prevent_initial_call=True  # allow_duplicate 때문에 True로 설정
)
def update_image_graphs_and_clear_shapes(selected_image, fig1, fig2):
    if selected_image is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # 환자 정보 가져오기
    patient_data = get_patient_info(selected_image)
    
    # 이미지의 중앙 슬라이스 계산
    axial_center = img.shape[0] // 2
    sagittal_center = img.shape[1] // 2
    
    # 병변이 있는 경우 시상면 최적 위치 조정
    if patient_data['hemorrhage_types']:
        sagittal_center = int(img.shape[1] * 0.45)
    
    print(f"🎯 그래프 업데이트 및 shapes 초기화: {selected_image}")
    print(f"   📍 최적 슬라이더 위치 - Axial: {axial_center}, Sagittal: {sagittal_center}")
    
    # 슬라이서 위치 설정
    slicer1.slice_idx = axial_center
    slicer2.slice_idx = sagittal_center
    
    # 현재 figure를 복사하고 shapes 초기화
    new_fig1 = fig1.copy() if fig1 else {}
    new_fig2 = fig2.copy() if fig2 else {}
    
    # layout이 없는 경우 생성
    if 'layout' not in new_fig1:
        new_fig1['layout'] = {}
    if 'layout' not in new_fig2:
        new_fig2['layout'] = {}
    
    # shapes 초기화
    new_fig1['layout']['shapes'] = []
    new_fig2['layout']['shapes'] = []
    
    # 기본 설정 다시 적용
    new_fig1['layout'].update({
        'dragmode': 'drawclosedpath',
        'newshape': {'line': {'color': 'cyan'}},
        'plot_bgcolor': 'rgb(0, 0, 0)'
    })
    
    new_fig2['layout'].update({
        'dragmode': 'drawrect',
        'newshape': {'line': {'color': 'cyan'}},
        'plot_bgcolor': 'rgb(0, 0, 0)'
    })
    
    print(f"   ✅ 그래프 업데이트 및 shapes 초기화 완료\n")
    
    return new_fig1, new_fig2, axial_center, sagittal_center

# 클라이언트 사이드 콜백 제거하고 서버 측 콜백으로 대체
@app.callback(
    Output("graph-helper", "figure"),
    [Input("occlusion-surface", "data")],
    [State("graph-helper", "figure")]
)
def update_3d_mesh(surf, current_figure):
    """3D 메쉬 업데이트를 위한 간단한 서버 측 콜백"""
    # surf가 None이면 현재 figure 유지
    if surf is None:
        return current_figure
    
    # 현재 figure에 surf 추가
    fig = current_figure.copy() if current_figure else fig_mesh
    
    # 기존 데이터가 있는 경우
    if 'data' in fig and len(fig['data']) > 0:
        # 기존 데이터가 두 개 이상이면 두 번째 데이터 교체
        if len(fig['data']) > 1:
            fig['data'][1] = surf
        else:
            # 데이터가 하나만 있는 경우 두 번째로 추가
            fig['data'].append(surf)
    else:
        # 데이터가 없는 경우 초기화
        fig['data'] = [surf]
    
    return fig

@app.callback(
    Output("annotations", "data"),
    [Input(slicer1.graph.id, "relayoutData"), Input(slicer2.graph.id, "relayoutData"),],
    [State("annotations", "data")],
)
def update_annotations(relayout1, relayout2, annotations):
    ctx = dash.callback_context
    # 아무 트리거도 발생하지 않았다면 업데이트하지 않음
    if not ctx.triggered:
        return dash.no_update
        
    if annotations is None:
        annotations = {}
    
    # relayout1 트리거에 의한 업데이트인 경우    
    if ctx.triggered[0]["prop_id"] == f"{slicer1.graph.id}.relayoutData":
        if relayout1 is not None and "shapes" in relayout1:
            if len(relayout1["shapes"]) >= 1:
                shape = relayout1["shapes"][-1]
                annotations["z"] = shape
            else:
                annotations.pop("z", None)
        elif relayout1 is not None and "shapes[2].path" in relayout1:
            if "z" in annotations:
                annotations["z"]["path"] = relayout1["shapes[2].path"]
    
    # relayout2 트리거에 의한 업데이트인 경우
    if ctx.triggered[0]["prop_id"] == f"{slicer2.graph.id}.relayoutData":
        if relayout2 is not None and "shapes" in relayout2:
            if len(relayout2["shapes"]) >= 1:
                shape = relayout2["shapes"][-1]
                annotations["x"] = shape
            else:
                annotations.pop("x", None)
        elif relayout2 is not None and (
            "shapes[2].y0" in relayout2 or "shapes[2].y1" in relayout2
        ):
            if "x" in annotations:
                if "shapes[2].y0" in relayout2:
                    annotations["x"]["y0"] = relayout2["shapes[2].y0"]
                if "shapes[2].y1" in relayout2:
                    annotations["x"]["y1"] = relayout2["shapes[2].y1"]
                    
    return annotations


# 슬라이더 변경 시 슬라이스 업데이트를 위한 콜백 추가
@app.callback(
    Output(slicer1.graph.id, "figure", allow_duplicate=True),
    [Input(slicer1.slider.id, "value")],
    prevent_initial_call=True
)
def update_axial_slice(slice_idx):
    if slice_idx is None or dash.callback_context.triggered_id != slicer1.slider.id:
        return dash.no_update
    
    # slice_idx 변경만으로 dash-slicer 내부적으로 slice가 변경됨
    # VolumeSlicer에서 직접 슬라이스 설정
    slicer1.slice_idx = slice_idx
    
    # 여기서는 dash.no_update를 반환하여 callback chain을 끊고
    # dash-slicer의 내부 로직이 figure를 업데이트하도록 함
    return dash.no_update

@app.callback(
    Output(slicer2.graph.id, "figure", allow_duplicate=True),
    [Input(slicer2.slider.id, "value")],
    prevent_initial_call=True
)
def update_sagittal_slice(slice_idx):
    if slice_idx is None or dash.callback_context.triggered_id != slicer2.slider.id:
        return dash.no_update
    
    # slice_idx 변경만으로 dash-slicer 내부적으로 slice가 변경됨
    # VolumeSlicer에서 직접 슬라이스 설정
    slicer2.slice_idx = slice_idx
    
    # 여기서는 dash.no_update를 반환하여 callback chain을 끊고
    # dash-slicer의 내부 로직이 figure를 업데이트하도록 함
    return dash.no_update

# 앱 시작 시 슬라이더 초기화를 위한 콜백 추가
@app.callback(
    [Output(slicer1.slider.id, "value", allow_duplicate=True),
     Output(slicer2.slider.id, "value", allow_duplicate=True)],
    [Input("patient-info", "className")],
    prevent_initial_call=True
)
def initialize_sliders(_):
    # 앱 시작 시 슬라이더 위치 초기화
    return axial_center, sagittal_center

@app.callback(
    [Output("graph-histogram", "figure"), Output("roi-warning", "is_open")],
    [Input("annotations", "data")],
)
def update_histo(annotations):
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        return dash.no_update, dash.no_update
    # Horizontal mask for the xy plane (z-axis)
    path = path_to_coords(annotations["z"]["path"])
    
    # 좌표를 이미지 경계 내로 제한하는 함수
    def clip_to_image_bounds(coords, shape):
        return np.clip(coords, 0, shape - 1)
    
    # 좌표 계산 및 경계 확인
    r_coords = path[:, 1] / spacing[1]
    c_coords = path[:, 0] / spacing[2]
    
    # 이미지 크기 가져오기
    height, width = img.shape[1], img.shape[2]
    
    # 폴리곤 좌표 생성 전에 경계 내에 있는지 확인
    if np.any(r_coords < 0) or np.any(r_coords >= height) or np.any(c_coords < 0) or np.any(c_coords >= width):
        print(f"경고: 일부 좌표가 이미지 경계를 벗어났습니다. 경계 내로 제한합니다.")
        r_coords = np.clip(r_coords, 0, height - 1)
        c_coords = np.clip(c_coords, 0, width - 1)
    
    # 폴리곤 좌표 생성
    try:
        rr, cc = draw.polygon(r_coords, c_coords)
        
        # 생성된 좌표가 경계 내에 있는지 다시 확인
        valid_indices = (rr < height) & (cc < width)
        if not np.all(valid_indices):
            print(f"경고: 생성된 폴리곤 좌표 중 {np.sum(~valid_indices)}개가 경계를 벗어났습니다.")
            rr = rr[valid_indices]
            cc = cc[valid_indices]
            
        if len(rr) == 0 or len(cc) == 0:
            print("오류: 유효한 폴리곤 좌표가 없습니다.")
            return dash.no_update, dash.no_update
                
        mask = np.zeros(img.shape[1:], dtype=bool)
        mask[rr, cc] = 1
        mask = ndimage.binary_fill_holes(mask)
        
    except Exception as e:
        print(f"폴리곤 생성 오류: {e}")
        return dash.no_update, dash.no_update
    
    # top and bottom, the top is a lower number than the bottom because y values
    # increase moving down the figure
    top, bottom = sorted([int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]])
    intensities = med_img[top:bottom, mask].ravel()
    if len(intensities) == 0:
        return dash.no_update, dash.no_update
    hi = exposure.histogram(intensities)
    fig = px.bar(
        x=hi[1],
        y=hi[0],
        # Histogram
        labels={"x": "HU 값", "y": "빈도"},
    )
    fig.update_layout(dragmode="select", title_font=dict(size=20, color="blue"))
    return fig, False

# 이미지 변경 시 레이아웃 자동 조정을 위한 서버 측 콜백
@app.callback(
    Output("patient-info", "className"),
    [Input("image-dropdown", "value")]
)
def adjust_layout_on_image_change(value):
    """이미지가 변경될 때 레이아웃을 자동으로 조정합니다."""
    # 여기서는 className만 반환하지만, 클라이언트에서 JavaScript로 레이아웃을 조정합니다.
    return "layout-adjusted"

# JavaScript 코드를 페이지에 삽입하여 레이아웃을 자동으로 조정
app.clientside_callback(
    """
    function(trigger) {
        if(trigger) {
            // autoscale 적용
            setTimeout(function() {
                var graphDivs = document.querySelectorAll('.js-plotly-plot');
                if(graphDivs && graphDivs.length > 0) {
                    for(var i=0; i < graphDivs.length; i++) {
                        try {
                            if(graphDivs[i] && graphDivs[i]._fullLayout) {
                                Plotly.relayout(graphDivs[i], {
                                    'xaxis.autorange': true,
                                    'yaxis.autorange': true
                                });
                            }
                        } catch(e) {
                            console.error("Plotly relayout error:", e);
                        }
                    }
                }
            }, 500);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("patient-info", "style"),
    [Input("patient-info", "className")]
)

@app.callback(
    [Output("occlusion-surface", "data"),
        Output(slicer1.overlay_data.id, "data"),
        Output(slicer2.overlay_data.id, "data"),
        Output("analysis-results", "children"),
        Output("infection-stats", "children"),
    ],
    [Input("graph-histogram", "selectedData"), Input("annotations", "data")],
)
def update_segmentation_slices(selected, annotations):
    ctx = dash.callback_context
    # When shape annotations are changed, reset segmentation visualization
    if (
        ctx.triggered[0]["prop_id"] == "annotations.data"
        or annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        # 이미지 크기에 맞는 빈 마스크 생성
        mask = np.zeros_like(med_img, dtype=bool)
        try:
            overlay1 = safe_create_overlay(slicer1, mask)
            overlay2 = safe_create_overlay(slicer2, mask)
        except Exception as e:
            print(f"안전한 오버레이 생성 실패: {e}")
            overlay1 = None
            overlay2 = None
        return go.Mesh3d(), overlay1, overlay2, "관심 영역을 선택하고 히스토그램에서 범위를 지정하세요.", "통계 정보가 여기에 표시됩니다."
    elif selected is not None and "range" in selected:
        if len(selected["points"]) == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        v_min, v_max = selected["range"]["x"]
        t_start = time()
        # Horizontal mask
        path = path_to_coords(annotations["z"]["path"])
        
        # 좌표 계산 및 경계 확인
        r_coords = path[:, 1] / spacing[1]
        c_coords = path[:, 0] / spacing[2]
        
        # 이미지 크기 가져오기
        height, width = img.shape[1], img.shape[2]
        
        # 폴리곤 좌표 생성 전에 경계 내에 있는지 확인
        if np.any(r_coords < 0) or np.any(r_coords >= height) or np.any(c_coords < 0) or np.any(c_coords >= width):
            print(f"경고: 일부 좌표가 이미지 경계를 벗어났습니다. 경계 내로 제한합니다.")
            r_coords = np.clip(r_coords, 0, height - 1)
            c_coords = np.clip(c_coords, 0, width - 1)
        
        # 폴리곤 좌표 생성
        try:
            rr, cc = draw.polygon(r_coords, c_coords)
            
            # 생성된 좌표가 경계 내에 있는지 다시 확인
            valid_indices = (rr < height) & (cc < width)
            if not np.all(valid_indices):
                print(f"경고: 생성된 폴리곤 좌표 중 {np.sum(~valid_indices)}개가 경계를 벗어났습니다.")
                rr = rr[valid_indices]
                cc = cc[valid_indices]
                
            if len(rr) == 0 or len(cc) == 0:
                print("오류: 유효한 폴리곤 좌표가 없습니다.")
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
                
            mask = np.zeros(img.shape[1:], dtype=bool)
            mask[rr, cc] = 1
            mask = ndimage.binary_fill_holes(mask)
            
        except Exception as e:
            print(f"폴리곤 생성 오류: {e}")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # top and bottom, the top is a lower number than the bottom because y values
        # increase moving down the figure
        top, bottom = sorted(
            [int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]]
        )
        
        # 이미지 높이 범위 확인 및 조정
        depth = img.shape[0]
        if top < 0 or bottom >= depth:
            print(f"경고: 높이 범위({top}, {bottom})가 이미지 높이({depth})를 벗어났습니다. 범위를 조정합니다.")
            top = max(0, min(top, depth-1))
            bottom = max(0, min(bottom, depth-1))
            if top >= bottom:
                bottom = min(top + 1, depth-1)
                
        # 마스크 생성
        img_mask = np.logical_and(med_img > v_min, med_img <= v_max)
        img_mask[:top] = False
        img_mask[bottom:] = False
        img_mask[top:bottom, np.logical_not(mask)] = False
        img_mask = largest_connected_component(img_mask)
        t_end = time()
        print("build the mask", t_end - t_start)
        
        t_start = time()
        # Update 3d viz
        try:
            # 최신 버전 API 시도
            verts, faces, _, _ = measure.marching_cubes(
                filters.median(img_mask, footprint=np.ones((1, 7, 7))), 0.5, step_size=3
            )
        except Exception as e:
            print(f"첫 번째 marching_cubes 오류: {e}")
            try:
                # 이전 버전 API 시도
                verts, faces, _, _ = measure.marching_cubes_lewiner(
                    volume=filters.median(img_mask, footprint=np.ones((1, 7, 7))), level=0.5, step_size=3
                )
            except Exception as e:
                print(f"두 번째 marching_cubes 오류: {e}")
                # 오류 발생 시 빈 메쉬 반환
                return go.Mesh3d(), safe_create_overlay(slicer1, img_mask), safe_create_overlay(slicer2, img_mask), "오류가 발생했습니다.", "통계를 계산할 수 없습니다."
        t_end = time()
        print("marching cubes", t_end - t_start)
        x, y, z = verts.T
        i, j, k = faces.T
        
        # 단순화된 3D 메쉬 생성 - 단일 Mesh3d 트레이스만 반환
        trace = go.Mesh3d(
            x=z, y=y, z=x, 
            color="red", 
            opacity=0.8, 
            i=k, j=j, k=i,
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                specular=0.8,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=100
            ),
            showscale=False
        )
        
        try:
            overlay1 = safe_create_overlay(slicer1, img_mask)
            overlay2 = safe_create_overlay(slicer2, img_mask)
        except Exception as e:
            print(f"안전한 오버레이 생성 실패: {e}")
            overlay1 = None
            overlay2 = None
        
        # 분석 결과 계산
        lesion_volume = np.sum(img_mask) * spacing[0] * spacing[1] * spacing[2]  # 입방 mm
        
        # 결과 텍스트 생성
        results = html.Div([
            html.P(f"선택된 HU 값 범위 : {v_min:.1f} - {v_max:.1f}"),
            html.P(f"병변 영역이 성공적으로 분할되었습니다."),
            html.P(f"3D 모델이 생성되었습니다. 오른쪽 패널에서 3D 시각화를 확인하세요."),
        ])
        
        # 출혈/경색 판단 로직 (HU 값 기준)
        if v_min >= 40:  # 일반적으로 급성 출혈은 HU 값이 높음
            ai_diagnosis = "급성 출혈 의심"
            ai_color = "red"
        elif v_max <= 40:  # 경색은 일반적으로 낮은 HU 값
            ai_diagnosis = "뇌경색 의심"
            ai_color = "blue"
        else:
            ai_diagnosis = "추가 검사 필요"
            ai_color = "green"
        
        # 현재 선택된 이미지의 환자 정보 가져오기 (교육용 비교)
        current_image = None
        for img_option in available_images:
            if 'value' in img_option:
                # 현재 분석 중인 이미지 찾기 (전역 변수 사용)
                try:
                    # 현재 로드된 환자 정보와 비교
                    patient_data = get_patient_info(img_option['value'])
                    if patient_data['total_slices'] > 0:  # 실제 환자 데이터가 있는 경우
                        current_image = img_option['value']
                        break
                except:
                    continue
        
        # 교육용 비교 정보
        education_content = []
        if current_image:
            patient_data = get_patient_info(current_image)
            actual_diagnosis = ', '.join(patient_data['hemorrhage_types']) if patient_data['hemorrhage_types'] else '정상'
            
            education_content.extend([
                html.Hr(),
                html.P([
                    html.Strong("AI 분석 결과 : "),
                    html.Span(ai_diagnosis, style={"color": ai_color})
                ], style={"marginBottom": "8px"}),
                html.P([
                    html.Strong("실제 방사학적 진단 : "),
                    html.Span(actual_diagnosis)
                ], style={"marginBottom": "8px"}),
                html.P([
                    html.Strong("학습 포인트 : "),
                    "HU 값만으로는 완전한 진단이 어려우며, 임상 소견과 함께 종합적으로 판단해야 합니다."
                ], style={"marginBottom": "8px", "fontSize": "0.9rem", "color": "black"})
            ])
        
        results = html.Div([
            html.P(f"선택된 HU 값 범위: {v_min:.1f} - {v_max:.1f}"),
            html.P(f"병변 영역이 성공적으로 분할되었습니다."),
            html.P(f"3D 모델이 생성되었습니다. 오른쪽 패널에서 3D 시각화를 확인하세요."),
            *education_content
        ])
        
        stats = html.Div([
            html.P([
                html.Strong("AI 분석 : ", style={"color": "black"}),
                html.Span(ai_diagnosis, style={"color": ai_color})
            ]),
            html.P([
                html.Strong("병변 부피 : ", style={"color": "black"}),
                html.Span(f"{lesion_volume:.1f} mm³")
            ]),
            html.P([
                html.Strong("슬라이스 범위 : ", style={"color": "black"}),
                html.Span(f"{top} - {bottom}")
            ]),
            html.P([
                html.Strong("관련 픽셀 수 : ", style={"color": "black"}),
                html.Span(f"{np.sum(img_mask)}")
            ]),
            
            # 교육용 추가 정보
            html.Hr(),
            html.H6("🔬 HU 값 기준", style={"fontSize": "0.95rem", "color": "black", "fontWeight": "bold"}),
            html.Ul([
                html.Li("급성 출혈: 50-90 HU"),
                html.Li("만성 출혈: 20-40 HU"), 
                html.Li("뇌경색: 10-30 HU"),
                html.Li("정상 뇌조직: 30-40 HU"),
                html.Li("뇌척수액: 0-15 HU")
            ], style={"fontSize": "0.85rem", "marginBottom": "10px", "lineHeight": "1.8"}),
            
            html.P([
                html.Strong("참고 : "),
                "HU(Hounsfield Unit) 값은 물의 밀도를 0으로 하는 상대적 척도입니다."
            ], className="text-muted", style={"fontSize": "0.8rem"})
        ])
        
        return trace, overlay1, overlay2, results, stats
    else:
        return (dash.no_update,) * 5

# 안전한 오버레이 생성 함수
def safe_create_overlay(slicer, mask):
    """크기 불일치를 처리하는 안전한 오버레이 생성 함수"""
    try:
        # 현재 슬라이서의 볼륨 크기 확인
        slicer_volume_shape = slicer._volume.shape
        
        # 마스크 크기가 다르면 조정
        if mask.shape != slicer_volume_shape:
            print(f"경고: 마스크 크기 {mask.shape}가 슬라이서 볼륨 크기 {slicer_volume_shape}와 다릅니다.")
            
            # 현재 이미지 크기에 맞는 새 마스크 생성
            adjusted_mask = np.zeros(slicer_volume_shape, dtype=bool)
            
            # 공통 크기 계산 (작은 쪽에 맞춤)
            min_z = min(mask.shape[0], slicer_volume_shape[0])
            min_y = min(mask.shape[1], slicer_volume_shape[1])
            min_x = min(mask.shape[2], slicer_volume_shape[2])
            
            # 유효한 영역만 복사
            adjusted_mask[:min_z, :min_y, :min_x] = mask[:min_z, :min_y, :min_x]
            mask = adjusted_mask
            print(f"마스크 크기를 {adjusted_mask.shape}로 조정했습니다.")
        
        return slicer.create_overlay_data(mask)
        
    except Exception as e:
        print(f"오버레이 생성 오류 ({slicer._axis}축): {e}")
        # 빈 마스크로 재시도
        try:
            empty_mask = np.zeros(slicer._volume.shape, dtype=bool)
            return slicer.create_overlay_data(empty_mask)
        except:
            print(f"빈 마스크 오버레이 생성도 실패했습니다.")
            return None

# 이미지 변경 시 어노테이션 초기화
@app.callback(
    Output("annotations", "data", allow_duplicate=True),
    [Input("image-dropdown", "value")],
    prevent_initial_call=True
)
def reset_annotations_on_image_change(selected_image):
    """이미지 변경 시 이전 어노테이션(마커) 초기화"""
    if selected_image is None:
        return dash.no_update
    print(f"🔄 이미지 변경으로 인한 어노테이션 초기화: {selected_image}")
    return {}

# 이미지 변경 시 히스토그램 선택 범위 초기화
@app.callback(
    Output("graph-histogram", "selectedData"),
    [Input("image-dropdown", "value")],
    prevent_initial_call=True
)
def reset_histogram_selection_on_image_change(selected_image):
    """이미지 변경 시 히스토그램에서 선택된 HU 값 범위 초기화"""
    if selected_image is None:
        return dash.no_update
    print(f"🔄 이미지 변경으로 인한 히스토그램 선택 범위 초기화: {selected_image}")
    return None

# 분석 컨텍스트 업데이트 콜백
@app.callback(
    Output("analysis-context", "data"),
    [Input("analysis-results", "children"),
     Input("infection-stats", "children"),
     Input("patient-info", "children"),
     Input("image-dropdown", "value")],  # 이미지 선택 변경 감지 추가
    prevent_initial_call=True
)
def update_analysis_context(analysis_results, infection_stats, patient_info, selected_image):
    """분석 결과를 챗봇 컨텍스트로 저장"""
    try:
        # 현재 선택된 이미지의 환자 정보 직접 가져오기
        if selected_image:
            patient_data = get_patient_info(selected_image)
        else:
            # 선택된 이미지가 없으면 첫 번째 이미지 사용
            current_image = available_images[0]['value'] if available_images else None
            patient_data = get_patient_info(current_image) if current_image else {}
        
        # 기본 컨텍스트 구성
        context = {
            "patient_number": patient_data.get('patient_num', '알수없음'),
            "age": patient_data.get('age', '알수없음'),
            "gender": patient_data.get('gender', '알수없음'),
            "diagnosis": ', '.join(patient_data.get('hemorrhage_types', [])) if patient_data.get('hemorrhage_types') else '정상',
            "fracture": patient_data.get('fracture', False),
            "detailed_diagnosis": patient_data.get('detailed_diagnosis', {}),
            "total_slices": patient_data.get('total_slices', 0),
            "affected_slices": patient_data.get('affected_slices', 0),
            "has_analysis": analysis_results is not None and str(analysis_results) != "관심 영역을 선택하고 히스토그램에서 범위를 지정하세요.",
            "timestamp": time(),
            "current_image": selected_image  # 현재 선택된 이미지 정보도 저장
        }
        
        # 분석이 완료된 경우 실제 분석 결과 파싱
        if context["has_analysis"] and analysis_results:
            try:
                # HTML 컴포넌트에서 텍스트 추출
                analysis_text = extract_text_from_html_component(analysis_results)
                
                # HU 값 범위 추출
                import re
                hu_match = re.search(r'선택된 HU 값 범위:\s*([-\d.]+)\s*-\s*([-\d.]+)', analysis_text)
                if hu_match:
                    context["actual_hu_range"] = {
                        "min": float(hu_match.group(1)),
                        "max": float(hu_match.group(2))
                    }
                
                # AI 분석 결과 추출
                ai_match = re.search(r'AI 분석 결과\s*:\s*([^\n\r]+)', analysis_text)
                if ai_match:
                    context["ai_analysis_result"] = ai_match.group(1).strip()
                
                # 실제 방사학적 진단 추출
                real_diagnosis_match = re.search(r'실제 방사학적 진단\s*:\s*([^\n\r]+)', analysis_text)
                if real_diagnosis_match:
                    context["real_diagnosis"] = real_diagnosis_match.group(1).strip()
                
                # 학습 포인트 추출
                learning_match = re.search(r'학습 포인트\s*:\s*([^\n\r]+)', analysis_text)
                if learning_match:
                    context["learning_point"] = learning_match.group(1).strip()
                
            except Exception as e:
                print(f"분석 결과 파싱 오류: {e}")
        
        # infection_stats(병변 통계)에서 추가 정보 추출
        if context["has_analysis"] and infection_stats:
            try:
                # HTML 컴포넌트에서 텍스트 추출
                stats_text = extract_text_from_html_component(infection_stats)
                
                # 병변 부피 추출
                volume_match = re.search(r'병변 부피\s*:\s*([\d.]+)\s*mm³', stats_text)
                if volume_match:
                    context["lesion_volume"] = float(volume_match.group(1))
                
                # 슬라이스 범위 추출
                slice_match = re.search(r'슬라이스 범위\s*:\s*(\d+)\s*-\s*(\d+)', stats_text)
                if slice_match:
                    context["slice_range"] = {
                        "start": int(slice_match.group(1)),
                        "end": int(slice_match.group(2))
                    }
                
                # 관련 픽셀 수 추출
                pixel_match = re.search(r'관련 픽셀 수\s*:\s*(\d+)', stats_text)
                if pixel_match:
                    context["related_pixels"] = int(pixel_match.group(1))
                
            except Exception as e:
                print(f"병변 통계 파싱 오류: {e}")
        
        print(f"🔄 챗봇 컨텍스트 업데이트: 환자 {context['patient_number']}, 분석 완료: {context['has_analysis']}")
        if context.get("real_diagnosis"):
            print(f"🔄 실제 진단: {context['real_diagnosis']}")
        if context.get("actual_hu_range"):
            print(f"   HU 범위: {context['actual_hu_range']['min']} - {context['actual_hu_range']['max']}")
        
        return context
    except Exception as e:
        print(f"분석 컨텍스트 업데이트 오류: {e}")
        return {}

# 챗봇 메시지 처리 콜백
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-input", "value"),
     Output("chat-history", "data")],
    [Input("chat-send-btn", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-history", "data"),
     State("analysis-context", "data")],
    prevent_initial_call=True
)
def handle_chat_message(send_clicks, input_submit, message, chat_history, analysis_context):
    """챗봇 메시지 처리"""
    if not message or message.strip() == "":
        return dash.no_update, dash.no_update, dash.no_update
    
    # 기본값 설정
    if chat_history is None:
        chat_history = []
    if analysis_context is None:
        analysis_context = {}
    
    # 사용자 메시지 추가
    user_message = {
        "type": "user",
        "content": message.strip(),
        "timestamp": time()
    }
    
    # 즉시 "생각중..." 메시지 표시
    thinking_message = {
        "type": "assistant",
        "content": "🤔 생각중...",
        "timestamp": time()
    }
    
    # 임시 히스토리 (생각중 메시지 포함)
    temp_history = chat_history + [user_message, thinking_message]
    
    try:
        # AI 응답 생성 - 실제 OpenAI API 사용
        print(f"🤖 AI 응답 생성 시작: {message}")
        ai_response = get_ai_response(message, analysis_context, chat_history)
        print(f"✅ AI 응답 완료: {len(ai_response)}자")
    except Exception as e:
        print(f"❌ AI 응답 생성 오류: {e}")
        ai_response = "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    # 실제 AI 응답으로 교체
    ai_message = {
        "type": "assistant",
        "content": ai_response,
        "timestamp": time()
    }
    
    # 최종 히스토리 (생각중 메시지 제거, 실제 응답 추가)
    new_history = chat_history + [user_message, ai_message]
    
    # 메시지 UI 컴포넌트 생성 - 자연스러운 채팅 버블 형태
    messages_components = []
    for msg in new_history:
        if msg["type"] == "user":
            # 사용자 메시지 - 우측 정렬, 파란색 버블
            messages_components.append(
                html.Div([
                    html.Div(
                        msg["content"],
                        style={
                            "backgroundColor": "#007bff", 
                            "color": "white",
                            "padding": "8px 12px",
                            "borderRadius": "18px 18px 4px 18px",
                            "display": "inline-block",
                            "maxWidth": "80%",
                            "wordWrap": "break-word",
                            "fontSize": "14px",
                            "lineHeight": "1.4"
                        }
                    )
                ], style={
                    "textAlign": "right",
                    "marginBottom": "8px",
                    "width": "100%",
                    "display": "block"
                })
            )
        else:
            # AI 메시지 - 좌측 정렬, 회색 버블
            messages_components.append(
                html.Div([
                    html.Div([
                        html.Span("🤖", style={"fontSize": "16px", "marginRight": "6px"}),
                        html.Span(
                            msg["content"],
                            style={
                                "whiteSpace": "pre-line"  # 줄바꿈 보존
                            }
                        )
                    ], style={
                        "backgroundColor": "#f1f1f1", 
                        "color": "#333",
                        "padding": "8px 12px",
                        "borderRadius": "18px 18px 18px 4px",
                        "display": "inline-block",
                        "maxWidth": "85%",
                        "wordWrap": "break-word",
                        "fontSize": "14px",
                        "lineHeight": "1.4"
                    })
                ], style={
                    "textAlign": "left",
                    "marginBottom": "8px",
                    "width": "100%",
                    "display": "block"
                })
            )
    
    return messages_components, "", new_history

def extract_text_from_html_component(component):
    """HTML 컴포넌트에서 텍스트만 추출하는 함수"""
    if component is None:
        return ""
    
    if isinstance(component, str):
        return component
    
    if isinstance(component, (int, float)):
        return str(component)
    
    # HTML 컴포넌트인 경우
    if hasattr(component, 'children'):
        return extract_text_from_html_component(component.children)
    
    # 딕셔너리 형태의 컴포넌트인 경우
    if isinstance(component, dict) and 'props' in component:
        props = component['props']
        if 'children' in props:
            return extract_text_from_html_component(props['children'])
        return ""
    
    # 리스트인 경우
    if isinstance(component, list):
        text_parts = []
        for item in component:
            text = extract_text_from_html_component(item)
            if text.strip():
                text_parts.append(text.strip())
        return " ".join(text_parts)
    
    return str(component) if component else ""

def update_analysis_context(analysis_results, infection_stats, patient_info, selected_image):
    """분석 결과를 챗봇 컨텍스트로 저장"""
    try:
        # 현재 선택된 이미지의 환자 정보 직접 가져오기
        if selected_image:
            patient_data = get_patient_info(selected_image)
        else:
            # 선택된 이미지가 없으면 첫 번째 이미지 사용
            current_image = available_images[0]['value'] if available_images else None
            patient_data = get_patient_info(current_image) if current_image else {}
        
        # 기본 컨텍스트 구성
        context = {
            "patient_number": patient_data.get('patient_num', '알수없음'),
            "age": patient_data.get('age', '알수없음'),
            "gender": patient_data.get('gender', '알수없음'),
            "diagnosis": ', '.join(patient_data.get('hemorrhage_types', [])) if patient_data.get('hemorrhage_types') else '정상',
            "fracture": patient_data.get('fracture', False),
            "detailed_diagnosis": patient_data.get('detailed_diagnosis', {}),
            "total_slices": patient_data.get('total_slices', 0),
            "affected_slices": patient_data.get('affected_slices', 0),
            "has_analysis": analysis_results is not None and str(analysis_results) != "관심 영역을 선택하고 히스토그램에서 범위를 지정하세요.",
            "timestamp": time(),
            "current_image": selected_image  # 현재 선택된 이미지 정보도 저장
        }
        
        # 분석이 완료된 경우 실제 분석 결과 파싱
        if context["has_analysis"] and analysis_results:
            try:
                # HTML 컴포넌트에서 텍스트 추출
                analysis_text = extract_text_from_html_component(analysis_results)
                
                # HU 값 범위 추출
                import re
                hu_match = re.search(r'선택된 HU 값 범위:\s*([-\d.]+)\s*-\s*([-\d.]+)', analysis_text)
                if hu_match:
                    context["actual_hu_range"] = {
                        "min": float(hu_match.group(1)),
                        "max": float(hu_match.group(2))
                    }
                
                # AI 분석 결과 추출
                ai_match = re.search(r'AI 분석 결과\s*:\s*([^\n\r]+)', analysis_text)
                if ai_match:
                    context["ai_analysis_result"] = ai_match.group(1).strip()
                
                # 실제 방사학적 진단 추출
                real_diagnosis_match = re.search(r'실제 방사학적 진단\s*:\s*([^\n\r]+)', analysis_text)
                if real_diagnosis_match:
                    context["real_diagnosis"] = real_diagnosis_match.group(1).strip()
                
                # 학습 포인트 추출
                learning_match = re.search(r'학습 포인트\s*:\s*([^\n\r]+)', analysis_text)
                if learning_match:
                    context["learning_point"] = learning_match.group(1).strip()
                
            except Exception as e:
                print(f"분석 결과 파싱 오류: {e}")
        
        # infection_stats(병변 통계)에서 추가 정보 추출
        if context["has_analysis"] and infection_stats:
            try:
                # HTML 컴포넌트에서 텍스트 추출
                stats_text = extract_text_from_html_component(infection_stats)
                
                # 병변 부피 추출
                volume_match = re.search(r'병변 부피\s*:\s*([\d.]+)\s*mm³', stats_text)
                if volume_match:
                    context["lesion_volume"] = float(volume_match.group(1))
                
                # 슬라이스 범위 추출
                slice_match = re.search(r'슬라이스 범위\s*:\s*(\d+)\s*-\s*(\d+)', stats_text)
                if slice_match:
                    context["slice_range"] = {
                        "start": int(slice_match.group(1)),
                        "end": int(slice_match.group(2))
                    }
                
                # 관련 픽셀 수 추출
                pixel_match = re.search(r'관련 픽셀 수\s*:\s*(\d+)', stats_text)
                if pixel_match:
                    context["related_pixels"] = int(pixel_match.group(1))
                
            except Exception as e:
                print(f"병변 통계 파싱 오류: {e}")
        
        print(f"🔄 챗봇 컨텍스트 업데이트: 환자 {context['patient_number']}, 분석 완료: {context['has_analysis']}")
        if context.get("real_diagnosis"):
            print(f"   실제 진단: {context['real_diagnosis']}")
        if context.get("actual_hu_range"):
            print(f"   HU 범위: {context['actual_hu_range']['min']} - {context['actual_hu_range']['max']}")
        
        return context
    except Exception as e:
        print(f"분석 컨텍스트 업데이트 오류: {e}")
        return {}

if __name__ == "__main__":
    # Render 배포를 위한 포트 및 호스트 설정
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    app.run_server(
        host=host,
        port=port, 
        debug=debug,
        dev_tools_props_check=False
    )