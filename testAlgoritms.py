import pypupilext as pp
import cv2
import numpy as np
import os
import re
import time
import pandas as pd

# ==========================================================
# 1. 알고리즘 선택 설정 (이 부분만 수정하세요)
# ==========================================================
ALGO_MAP = {
    'PuRe': pp.PuRe,
    'PuReST': pp.PuReST,
    'ElSe': pp.ElSe,
    'ExCuSe': pp.ExCuSe,
    'Starburst': pp.Starburst,
    'Swirski2D': pp.Swirski2D
}

# ==========================================================
# 1. 설정
# ==========================================================
CONFIG = {
    'SELECTED_ALGO': 'PuReST',
    # 'INPUT_PATH': 'D:/Github/EyeTracking/Datasets/Open-PupilEXT/Demo_Data_1_Single_Camera.gif', # 동영상 예시
    'INPUT_PATH': 'D:/Github/EyeTracking/Datasets/Swirski/p2-right/frames/', # 폴더 예시
    'VIEW_WIDTH': 400,
    'GRAPH_WIDTH': 600,
    'GRAPH_HEIGHT': 300 
}

def natural_sort_key(s):
    """숫자가 포함된 파일명을 숫자 크기순으로 정렬 (frame1, frame2, ... frame100000)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class PupilTracker:
    def __init__(self, config):
        self.config = config
        self.algo_name = config['SELECTED_ALGO']
        algo_class = ALGO_MAP.get(self.algo_name, pp.PuRe) # 기본값 PuRe
        self.algorithm = algo_class()
        print(f"[*] Selected Algorithm: {self.algo_name}")
        
        self.results = []
        self.paused = False
        self.frame_idx = 0  # 이미지 폴더 처리를 위한 인덱스
        self.data_records = []  # CSV 저장을 위한 데이터 보관용 리스트
        
        # 파일 저장 경로 설정 (입력 파일명 기반)
        input_basename = os.path.basename(config['INPUT_PATH']).split('.')[0]
        print(input_basename)
        self.output_filename = f"Results_{self.algo_name}_{input_basename}.csv"

        # 데이터 히스토리
        self.history_conf = []
        self.history_diam = []
        self.history_x = []
        self.history_y = []

        self._init_source()
        self._setup_windows()

    def _init_source(self):
        input_path = self.config['INPUT_PATH']
        self.is_video = not os.path.isdir(input_path)
        
        if self.is_video:
            # 1. 동영상 소스 초기화
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps_delay = int(1000 / fps) if fps > 0 else 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[Video Mode] Res: {self.fw}x{self.fh}, FPS: {fps}, Total: {self.total_frames}")
        else:
            # 2. 이미지 폴더 소스 초기화
            self.image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.image_files.sort(key=natural_sort_key)
            self.total_frames = len(self.image_files)
            
            if self.total_frames == 0:
                raise ValueError(f"No images found in folder: {input_path}")
            
            # 첫 번째 이미지에서 해상도 정보 가져오기
            first_img = cv2.imread(os.path.join(input_path, self.image_files[0]))
            self.fh, self.fw = first_img.shape[:2]
            self.fps_delay = 10 # 폴더 재생 시 기본 딜레이
            print(f"[Folder Mode] Total Images: {self.total_frames}, Sample Res: {self.fw}x{self.fh}")

    def _setup_windows(self):
        """알고리즘 특성에 맞게 필요한 제어창만 생성합니다."""
        # 그래프 캔버스 초기화 (이건 항상 필요)
        self.graph_canvas = np.zeros((self.config['GRAPH_HEIGHT'], self.config['GRAPH_WIDTH'], 3), dtype=np.uint8)

        # 알고리즘이 물리적 직경 설정을 지원하는 경우에만 제어창 생성
        if hasattr(self.algorithm, 'maxPupilDiameterMM'):
            cv2.namedWindow("Control Panel")
            cv2.createTrackbar("Max Diameter (mm)", "Control Panel", 7, 20, lambda x: None)
            print(f"[*] '{self.algo_name}' supports physical constraints. Control Panel enabled.")
        else:
            print(f"[*] '{self.algo_name}' does not require physical constraints. Control Panel hidden.")

    def _draw_realtime_graph(self, confidence, diameter, x, y):
        # (기존 코드와 동일하므로 생략 - 리스트 관리 및 그리기 로직)
        self.history_conf.append(confidence)
        self.history_diam.append(diameter)
        self.history_x.append(x)
        self.history_y.append(y)
        
        max_pts = self.config['GRAPH_WIDTH']
        if len(self.history_conf) > max_pts:
            for attr in [self.history_conf, self.history_diam, self.history_x, self.history_y]:
                attr.pop(0)

        self.graph_canvas.fill(20) 
        mid_y = self.config['GRAPH_HEIGHT'] // 2
        cv2.line(self.graph_canvas, (0, mid_y), (max_pts, mid_y), (100, 100, 100), 1)

        for i in range(1, len(self.history_conf)):
            # 상단: Confidence & Diameter
            cv2.line(self.graph_canvas, 
                     (i-1, mid_y - int(self.history_conf[i-1] * mid_y)),
                     (i, mid_y - int(self.history_conf[i] * mid_y)), (255, 100, 0), 1)
            cv2.line(self.graph_canvas, 
                     (i-1, mid_y - int(self.history_diam[i-1] % mid_y)),
                     (i, mid_y - int(self.history_diam[i] % mid_y)), (0, 255, 0), 1)

            # 하단: X & Y
            scale_y = lambda val, max_val: int(mid_y + (val / max_val) * mid_y)
            cv2.line(self.graph_canvas, 
                     (i-1, scale_y(self.history_x[i-1], self.fw)),
                     (i, scale_y(self.history_x[i], self.fw)), (0, 0, 255), 1)
            cv2.line(self.graph_canvas, 
                     (i-1, scale_y(self.history_y[i-1], self.fh)),
                     (i, scale_y(self.history_y[i], self.fh)), (0, 255, 255), 1)

        cv2.putText(self.graph_canvas, f"Blue:Conf / Green:Diam({diameter:.1f})", (10, 20), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(self.graph_canvas, f"Red:X / Yellow:Y", (10, mid_y + 20), 0, 0.4, (255, 255, 255), 1)
        cv2.imshow("Real-time Stats", self.graph_canvas)

    def run(self):
        while True:
            if not self.paused:
                frame = None
                if self.is_video:
                    ret, frame = self.cap.read()
                    if not ret: break
                else:
                    if self.frame_idx >= self.total_frames: break
                    img_path = os.path.join(self.config['INPUT_PATH'], self.image_files[self.frame_idx])
                    frame = cv2.imread(img_path)
                    self.frame_idx += 1
                
                # 분석 프로세스 (공통)
                if hasattr(self.algorithm, 'maxPupilDiameterMM'):
                    val = cv2.getTrackbarPos("Max Diameter (mm)", "Control Panel")
                    self.algorithm.maxPupilDiameterMM = val
                    if val != -1:
                        self.algorithm.maxPupilDiameterMM = val

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                start_time = time.perf_counter()  # 측정 시작
                try:
                    pupil = self.algorithm.runWithConfidence(gray)
                except AttributeError:
                    # runWithConfidence가 없는 구형 알고리즘 대응
                    pupil = self.algorithm.run(gray)
                end_time = time.perf_counter()    # 측정 종료
                processing_time_ms = (end_time - start_time) * 1000

                # 3. 데이터 기록 (프레임 인덱스, 좌표, 직경, 신뢰도 등)
                current_data = {
                    'Frame': self.frame_idx,
                    'Algo': self.algo_name,
                    'X': round(pupil.center[0], 2),
                    'Y': round(pupil.center[1], 2),
                    'Diameter': round(pupil.diameter(), 4),
                    'Confidence': round(pupil.outline_confidence, 4),
                    'MajorAxis': round(pupil.majorAxis(), 4),
                    'MinorAxis': round(pupil.minorAxis(), 4),
                    'Angle': round(pupil.angle, 2)
                }
                self.data_records.append(current_data)

                self._draw_realtime_graph(pupil.outline_confidence, pupil.diameter(), 
                                          pupil.center[0], pupil.center[1])

                self._draw_info(frame, pupil)
                
                # 진행률 표시 (추가)
                progress = (self.frame_idx if not self.is_video else self.cap.get(cv2.CAP_PROP_POS_FRAMES)) / self.total_frames * 100
                cv2.putText(frame, f"{self.algo_name}", (10, 30), 0, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 60), 0, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Proc: {processing_time_ms:.2f}ms", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                
                view_frame = cv2.resize(frame, (self.config['VIEW_WIDTH'], int(frame.shape[0] * (self.config['VIEW_WIDTH']/frame.shape[1]))))
                cv2.imshow("Tracking", view_frame)

            key = cv2.waitKey(self.fps_delay if not self.paused else 1) & 0xFF
            if key == ord('q'): break
            if key == ord(' '): self.paused = not self.paused

        # 분석 종료 후 저장 프로세스 호출
        self._save_to_csv()

        print("[*] Analysis complete. Closing...")
        # print("Press any key to exit.")
        # cv2.waitKey(0) # 키 입력 시까지 정지
        # if self.is_video: self.cap.release()
        cv2.destroyAllWindows()

    def _save_to_csv(self):
        """기록된 데이터를 CSV 파일로 저장합니다."""
        if not self.data_records:
            print("[!] No data to save.")
            return

        try:
            df = pd.DataFrame(self.data_records)
            df.to_csv(self.output_filename, index=False, encoding='utf-8-sig')
            print(f"[+] Successfully saved results to '{self.output_filename}'")
        except Exception as e:
            print(f"[!] Error saving CSV: {e}")

    def _draw_info(self, frame, pupil):
        # 1. 장축(Major)을 먼저 인자로 넣어야 하는 알고리즘 그룹 정의
        MAJOR_FIRST_ALGOS = ['Swirski2D', 'Starburst']
        # 2. 알고리즘 이름에 따라 축 순서 결정
        if self.algo_name in MAJOR_FIRST_ALGOS:
            axes = (int(pupil.majorAxis() / 2), int(pupil.minorAxis() / 2))
        else:
            # PuRe, ElSe, ExCuSe 등은 일반적으로 단축을 먼저 넣거나 
            # pypupilext의 기본 출력 형식을 따름
            axes = (int(pupil.minorAxis() / 2), int(pupil.majorAxis() / 2))

        # 3. 타원 그리기 (두께 1로 설정하신 값 반영)
        cv2.ellipse(frame, 
                    (int(pupil.center[0]), int(pupil.center[1])),
                    axes, 
                    pupil.angle, 0, 360, (0, 255, 0), 1)

        cv2.circle(frame, (int(pupil.center[0]), int(pupil.center[1])), 2, (0, 0, 255), -1)

if __name__ == "__main__":
    tracker = PupilTracker(CONFIG)
    tracker.run()