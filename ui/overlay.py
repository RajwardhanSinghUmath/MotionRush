import cv2
import time

class UIOverlay:
    def __init__(self):
        self.prev_time = time.monotonic()
        self.connections = [
            (11, 12),           # shoulders
            (11, 13), (13, 15), # left arm
            (12, 14), (14, 16), # right arm
            (11, 23), (12, 24), # torso
            (23, 24),           # hips
            (23, 25), (25, 27), # left leg
            (24, 26), (26, 28)  # right leg
        ]
        self.action_display_time = 0
        self.last_action_text = ""

    def notify_action(self, action_text):
        if action_text and action_text not in ["None", "Cooldown"]:
            self.last_action_text = action_text
            self.action_display_time = time.monotonic()

    def draw_skeleton(self, frame, results):
        if results and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            height, width, _ = frame.shape
            
            # Draw skeleton bones
            for connection in self.connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                        start_px = (int(start_point.x * width), int(start_point.y * height))
                        end_px = (int(end_point.x * width), int(end_point.y * height))
                        cv2.line(frame, start_px, end_px, (180, 100, 80), 2)
            
            # Draw joints
            for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        px = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(frame, px, 4, (60, 180, 220), -1)
                        
        return frame

    def draw_boundary_markers(self, frame, recognizer):
        h, w, _ = frame.shape
        if not recognizer.is_calibrated:
            cv2.rectangle(frame, (round(w * 0.05), round(h * 0.05)),
                          (round(w * 0.95), round(h * 0.95)), (120, 120, 120), 1)
            return frame
        y_upper_px = int(recognizer.y_upper * h)
        y_lower_px = int(recognizer.y_lower * h)
        x_left_px = int(recognizer.x_left * w)
        x_right_px = int(recognizer.x_right * w)

        # -------------------------------------------------------------
        # 1. TWO HORIZONTAL LINES (UP / DOWN)
        # -------------------------------------------------------------
        # Upper Line (Jump)
        is_up = (recognizer.vertical_state == "UP")
        upper_color = (0, 255, 255) if is_up else (60, 210, 220)
        cv2.line(frame, (0, y_upper_px), (w, y_upper_px), upper_color, 2 if not is_up else 3)
        cv2.putText(frame, "^ JUMP LINE (UP) ^", (15, y_upper_px - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, upper_color, 1, cv2.LINE_AA)

        # Lower Line (Squat)
        is_down = (recognizer.vertical_state == "DOWN")
        lower_color = (0, 140, 255) if is_down else (60, 140, 200)
        cv2.line(frame, (0, y_lower_px), (w, y_lower_px), lower_color, 2 if not is_down else 3)
        # Place label right above the lower line so it stays visible and clean
        cv2.putText(frame, "v SQUAT LINE (DOWN) v", (15, y_lower_px - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lower_color, 1, cv2.LINE_AA)

        # Neutral Zone Label in Center
        neutral_y = int((y_upper_px + y_lower_px) / 2)
        cv2.putText(frame, "[ NEUTRAL - NO KEY ]", (15, neutral_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. TWO VERTICAL LINES (LEFT / RIGHT LANES)
        # -------------------------------------------------------------
        # Left Lane Line
        is_left = (recognizer.current_lane == "LEFT")
        left_color = (0, 255, 0) if is_left else (150, 150, 150)
        cv2.line(frame, (x_left_px, 0), (x_left_px, h), left_color, 2)
        cv2.putText(frame, "< LEFT", (x_left_px - 60, int(h * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, left_color, 1, cv2.LINE_AA)

        # Right Lane Line
        is_right = (recognizer.current_lane == "RIGHT")
        right_color = (0, 255, 0) if is_right else (150, 150, 150)
        cv2.line(frame, (x_right_px, 0), (x_right_px, h), right_color, 2)
        cv2.putText(frame, "RIGHT >", (x_right_px + 10, int(h * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, right_color, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. WAIST MARKER
        # -------------------------------------------------------------
        if recognizer.waist_position:
            wx, wy = recognizer.waist_position
            w_px = (int(wx * w), int(wy * h))
            
            # Dynamic marker styling
            if is_up:
                marker_color = (0, 255, 255) # Yellow
            elif is_down:
                marker_color = (0, 140, 255) # Orange
            elif recognizer.current_lane != "CENTER":
                marker_color = (0, 255, 0)   # Green
            else:
                marker_color = (0, 230, 255) # Cyan

            # Outer ring & center dot
            cv2.circle(frame, w_px, 13, marker_color, 2)
            cv2.circle(frame, w_px, 4, marker_color, -1)
            # Crosshairs
            cv2.line(frame, (w_px[0] - 16, w_px[1]), (w_px[0] + 16, w_px[1]), marker_color, 1)
            cv2.line(frame, (w_px[0], w_px[1] - 16), (w_px[0], w_px[1] + 16), marker_color, 1)
            cv2.putText(frame, f"WAIST (y={wy:.2f})", (w_px[0] + 16, w_px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, marker_color, 1, cv2.LINE_AA)

        return frame

    def draw_hud(self, frame, recognizer):
        now = time.monotonic()
        fps = 1.0 / max(0.001, (now - self.prev_time))
        self.prev_time = now

        h, w, _ = frame.shape

        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 36), (25, 25, 25), -1)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Status text
        lane_str = f"LANE: [{recognizer.current_lane}]"
        vert_str = f"VERT: [{recognizer.vertical_state}]"
        cv2.putText(frame, lane_str, (95, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, vert_str, (245, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)

        if not recognizer.is_calibrated or recognizer.waist_position is None:
            # Wrap the player instruction to keep it readable in the preview.
            message = recognizer.status_message
            instruction_scale = w / 640
            font_size = 0.85 * instruction_scale
            line_height = max(20, round(34 * instruction_scale))
            words = message.split()
            lines = ['']
            for word in words:
                candidate = (lines[-1] + ' ' + word).strip()
                if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0][0] > w - 24:
                    lines.append(word)
                else:
                    lines[-1] = candidate
            cv2.rectangle(frame, (0, 38), (w, 48 + line_height * len(lines)), (25, 25, 25), -1)
            for index, line in enumerate(lines):
                cv2.putText(frame, line, (12, 38 + line_height * (index + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 210, 255), 1, cv2.LINE_AA)

        # Action Flash Banner
        if recognizer.is_calibrated and (now - self.action_display_time) < 0.6:
            action_text = f">> {self.last_action_text.upper()} <<"
            cv2.rectangle(frame, (int(w*0.20), 44), (int(w*0.80), 80), (0, 200, 0), -1)
            cv2.putText(frame, action_text, (int(w*0.23), 69),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)

        # Compact hotkeys bottom footer bar
        cv2.rectangle(frame, (0, h - 22), (w, h), (18, 18, 18), -1)
        controls_text = "[C] Calibrate | [[ / ]] Gap | [W/S] Lines | [Q] Quit"
        cv2.putText(frame, controls_text, (8, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

        return frame
