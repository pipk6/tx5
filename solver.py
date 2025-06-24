import requests
import json
import cv2
import numpy as np
import os
import time
import uuid
import threading
import logging

# 建议在您的主程序中配置 logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

class CaptchaSolver:
    """
    一个线程安全的、使用 Session 的验证码解决程序。

    此类可被多线程安全地共享。它使用 requests.Session 管理会话和 Cookies，
    并为每次解算尝试使用唯一的临时文件以避免竞争条件。
    内置了请求重试机制。
    """

    def __init__(self, aid="189981187", max_retries=6):
        """
        初始化验证码解决程序。

        Args:
            aid (str): 验证码服务的应用 ID。
            max_retries (int): 每个网络请求的最大重试次数。
        """
        self.aid = aid
        self.max_retries = max_retries
        # 将临时图片目录指向保证可写的 /tmp 目录
        self.img_dir = '/tmp/img'

        # 确保图片目录存在
        # 使用 exist_ok=True 避免多线程同时创建目录时出错
        os.makedirs(self.img_dir, exist_ok=True)

    def _make_request(self, session, method, url, **kwargs):
        """
        使用 session 并带有重试逻辑的请求辅助方法。
        """
        retries = 0
        while retries < self.max_retries:
            try:
                # 移除了代理逻辑
                if method.lower() == 'get':
                    response = session.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = session.post(url, **kwargs)
                else:
                    raise ValueError("不支持的 HTTP 方法")

                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logging.warning(f"请求错误: {e}, 正在重试 ({retries + 1}/{self.max_retries})...")
                retries += 1
                time.sleep(1)

        logging.error(f"达到最大重试次数，请求失败: {url}")
        return None

    def solve(self):
        """
        尝试解决验证码，此方法是线程安全的。
        """
        session = requests.Session()
        # --- 改进 #1: 添加更详细的请求头，模拟真实浏览器 ---
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Referer": "https://ca.turing.captcha.qcloud.com/", # 添加一个通用的 Referer
            "X-Requested-With": "XMLHttpRequest",
        })

        unique_id = uuid.uuid4()
        thread_name = threading.current_thread().name
        template_path = os.path.join(self.img_dir, f"captcha_template_{thread_name}_{unique_id}.png")
        target_path = os.path.join(self.img_dir, f"captcha_target_{thread_name}_{unique_id}.png")

        try:
            for attempt in range(1, self.max_retries):
                logging.info(f"开始第 {attempt}/{self.max_retries} 次解算尝试...")
                try:
                    captcha_data = self._fetch_captcha_data(session, target_path, template_path)
                    if not captcha_data:
                        logging.warning("获取验证码数据失败，进入下一次尝试...")
                        continue

                    answer_payload = self._process_images(target_path, template_path)
                    if not answer_payload:
                        logging.warning("图像处理失败，进入下一次尝试...")
                        continue

                    verification_result = self._verify_captcha(session, captcha_data['session_id'], answer_payload)

                    logging.info(f"验证响应: {verification_result}")

                    if verification_result and verification_result.get('errorCode') == '0':
                        logging.info("验证码成功解决!")
                        randstr = verification_result.get("randstr")
                        ticket = verification_result.get("ticket")
                        return randstr, ticket
                    # else:
                    #     if verification_result:
                    #         err_code = verification_result.get('errorCode')
                    #         err_msg = verification_result.get('errMessage', 'No error message provided.')
                    #         logging.error(f"验证失败。错误码: {err_code}, 错误信息: {err_msg}")
                    #     else:
                    #         logging.error("验证失败，未收到有效的响应。")


                except Exception as e:
                    logging.error(f"第 {attempt} 次尝试中发生未知异常: {e}", exc_info=True)

                if attempt < 3:
                    time.sleep(1)

            logging.error("所有解算尝试均失败。")
            return None
        finally:
            if os.path.exists(template_path): os.remove(template_path)
            if os.path.exists(target_path): os.remove(target_path)

    def _fetch_captcha_data(self, session, target_path, template_path):
        url = "https://ca.turing.captcha.qcloud.com/cap_union_prehandle"
        params = {"aid": self.aid}

        initial_response = self._make_request(session, 'get', url, params=params, timeout=10)
        if not initial_response: return None

        try:
            raw_text = initial_response.text
            json_str = raw_text[raw_text.find('(') + 1 : raw_text.rfind(')')]
            response_data = json.loads(json_str)

            base_url = "https://ca.turing.captcha.qcloud.com"
            bg_img_url = base_url + response_data['data']['dyn_show_info']['bg_elem_cfg']['img_url']
            sprite_img_url = base_url + response_data['data']['dyn_show_info']['sprite_url']

            bg_response = self._make_request(session, 'get', bg_img_url, timeout=10)
            if bg_response:
                with open(target_path, 'wb') as f: f.write(bg_response.content)
            else: return None

            sprite_response = self._make_request(session, 'get', sprite_img_url, timeout=10)
            if sprite_response:
                with open(template_path, 'wb') as f: f.write(sprite_response.content)
            else: return None

            return {"session_id": response_data['sess']}

        except (KeyError, json.JSONDecodeError, AttributeError) as e:
            logging.error(f"解析验证码数据时出错: {e}")
            return None

    def _verify_captcha(self, session, session_id, answer):
        url = 'https://ca.turing.captcha.qcloud.com/cap_union_new_verify'
        answer_str = json.dumps(answer, separators=(',', ':'))
        payload_str = f'sess={session_id}&ans={answer_str}'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        response = self._make_request(session, 'post', url, data=payload_str, headers=headers, timeout=10)

        if response:
            try:
                return response.json()
            except json.JSONDecodeError:
                logging.error(f"验证时解码JSON响应失败: {response.text}")
                return None
        return None

    def _process_images(self, target_path, template_path):
        detected_symbols = self._detect_black_symbols(target_path)
        if not detected_symbols: return None
        symbol_centers = self._locate_symbol_positions(template_path, target_path)
        if not symbol_centers or len(symbol_centers) < 3:
            if len(detected_symbols) >= 3:
                symbol_centers = [(s[0] + s[2] // 2, s[1] + s[3] // 2) for s in detected_symbols[:3]]
            else: return None
        matched_symbols, remaining_symbols = [], list(detected_symbols)
        for center in symbol_centers:
            closest_idx, symbol, distance = self._find_closest_symbol(center, remaining_symbols)
            if symbol and distance < 100:
                matched_symbols.append(symbol)
                if closest_idx < len(remaining_symbols): remaining_symbols.pop(closest_idx)
            else:
                w, h = 50, 50
                matched_symbols.append((int(center[0] - w/2), int(center[1] - h/2), w, h))
        result_data = []
        for idx, (x, y, w, h) in enumerate(matched_symbols):
            result_data.append({"elem_id": idx + 1, "type": "DynAnswerType_POS", "data": f"{x + w // 2},{y + h // 2}"})
        return result_data

    def _detect_black_symbols(self, image_path, min_area=500, max_area=5000):
        target_image = cv2.imread(image_path)
        if target_image is None: return []
        hsv_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0]); upper_black = np.array([180, 255, 70])
        mask = cv2.inRange(hsv_image, lower_black, upper_black)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_symbols = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                if 0.2 < float(w) / h < 2.0: detected_symbols.append((x, y, w, h))
        detected_symbols.sort(key=lambda s: s[0])
        return detected_symbols

    def _locate_symbol_positions(self, template_path, target_path):
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None or target_img is None: return []
        h, w = template_img.shape
        sw = w // 3
        templates = [template_img[:, i*sw:(i+1)*sw] for i in range(3)]
        centers = []
        for t in templates:
            center = self._match_with_sift(t, target_img) or self._match_with_template(t, target_img)
            centers.append(center)
        return [c for c in centers if c is not None]

    def _match_with_sift(self, template, target_image):
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(target_image, None)
            if des1 is None or des2 is None or len(kp1) < 4: return None
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            # --- 改进 #3: 稍微收紧匹配阈值 ---
            matches = [m for m, n in flann.knnMatch(des1, des2, k=2) if m.distance < 0.8 * n.distance]
            if len(matches) < 4: return None
            pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            return (int(np.median(pts[:, 0])), int(np.median(pts[:, 1])))
        except Exception: return None

    def _match_with_template(self, template, target_image):
        try:
            res = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < 0.6: return None
            h, w = template.shape
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
        except Exception: return None

    def _find_closest_symbol(self, center, symbols):
        if not symbols: return -1, None, float('inf')
        min_dist = float('inf')
        closest_idx = -1
        for idx, (x, y, w, h) in enumerate(symbols):
            dist = np.sqrt((center[0] - (x + w/2))**2 + (center[1] - (y + h/2))**2)
            if dist < min_dist: min_dist = dist; closest_idx = idx
        return closest_idx, symbols[closest_idx], min_dist
