import pypupilext as pp
import cv2
import pandas as pd
import os
import re

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None: return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

# ==========================================================
# 1. 알고리즘 선택 설정 (이 부분만 수정하세요)
# ==========================================================
SELECTED_ALGO = 'PuReST'  # 원하는 알고리즘으로 변경
ALGO_MAP = {
    'PuRe': pp.PuRe,
    'PuReST': pp.PuReST,
    'ElSe': pp.ElSe,
    'ExCuSe': pp.ExCuSe,
    'Starburst': pp.Starburst,
    'Swirski2D': pp.Swirski2D
}

# 알고리즘 인스턴스 생성
algorithm = ALGO_MAP[SELECTED_ALGO]()

# 알고리즘별 특화 파라미터 설정 (필요 시)
if SELECTED_ALGO == 'PuRe':
    algorithm.maxPupilDiameterMM = 7
# ==========================================================

def natural_sort_key(s):
    """숫자가 포함된 파일명을 순서대로 정렬하기 위한 키 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

#input_path = 'D:/Github/EyeTracking/Datasets/Swirski/p2-left/frames/'
input_path = 'D:/Github/EyeTracking/Datasets/Open-PupilEXT/Demo_Data_2_Single_Camera.gif'

results = []

is_video = not os.path.isdir(input_path)
# 1. 입력 소스 초기화
if not is_video:
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg'))]
    image_files.sort(key=natural_sort_key)
    total_frames = len(image_files)
    fps_delay = 30
else:
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_delay = int(1000 / fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"해상도: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")
    

print(f"[{SELECTED_ALGO}] 엔진으로 {total_frames}개의 이미지를 분석합니다.")

paused = False
frame_idx = 0

while True:
    if not paused:
        # 프레임 읽기
        if not is_video:
            if frame_idx >= total_frames: break
            filename = image_files[frame_idx]
            frame = cv2.imread(os.path.join(input_path, filename))
        else:
            ret, frame = cap.read()
            if not ret: break
            filename = f"frame_{frame_idx}"

        if frame is None: break
        
    # 동공 검출 실행
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil = algorithm.runWithConfidence(gray)
        
        results.append({
            'FileName': filename,
            'Algo': SELECTED_ALGO,
            'Outline Conf': pupil.outline_confidence,
            'PupilDiameter': pupil.diameter(),
            'CenterX': pupil.center[0],
            'CenterY': pupil.center[1]
        })

        # 실시간 시각화
        if (SELECTED_ALGO=='Swirski2D') or (SELECTED_ALGO == 'Starburst'):
            cv2.ellipse(frame,
                        (int(pupil.center[0]), int(pupil.center[1])),
                        (int(pupil.majorAxis()/2), int(pupil.minorAxis()/2)), 
                        pupil.angle, 0, 360, (0, 255, 0), 1)
        else:
            cv2.ellipse(frame,
                        (int(pupil.center[0]), int(pupil.center[1])),
                        (int(pupil.minorAxis()/2), int(pupil.majorAxis()/2)), 
                        pupil.angle, 0, 360, (0, 255, 0), 1)      
        cv2.circle(frame,
                    (int(pupil.center[0]), int(pupil.center[1])),
                    2, 
                    (0, 0, 255), 2)
        # 알고리즘 이름을 화면에 표시
        cv2.putText(frame, f"Algo: {SELECTED_ALGO}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        resize = ResizeWithAspectRatio(frame, width=400)
        cv2.imshow("Pupil Tracking Test", resize)

        frame_idx += 1
    except Exception as e:
        print(f"Error processing {filename}: {e}")

    # 키 입력 처리
    key = cv2.waitKey(fps_delay if not paused else 1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('u') and CONFIG['METHOD_TYPE'] == 'STATIC':
        current_bw_threshold += 5
        print(f"Threshold: {current_bw_threshold}")
    elif key == ord('d') and CONFIG['METHOD_TYPE'] == 'STATIC':
        current_bw_threshold = max(current_bw_threshold - 5, 0)
        print(f"Threshold: {current_bw_threshold}")

if is_video:
    cap.release()

df = pd.DataFrame(results)
print(f"\n--- [{SELECTED_ALGO}] 분석 완료 ---")
print(df.head())

# 결과 저장 (파일명에 알고리즘 이름 포함)
# df.to_csv(f'results_{SELECTED_ALGO}.csv', index=False)

cv2.destroyAllWindows()