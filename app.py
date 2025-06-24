import os
import asyncio
import logging
import time
import collections
from quart import Quart, request, jsonify
from quart_cors import cors
from hypercorn.asyncio import serve
from hypercorn.config import Config
from solver import CaptchaSolver

# --- 配置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- 从环境变量中获取 VIP 码 ---
vip_codes_str = os.environ.get('VIP_CODES')
if vip_codes_str:
    vipCodes = vip_codes_str.split(',')
    logging.info(f"成功从环境变量加载了 {len(vipCodes)} 个 VIP 码。")
else:
    vipCodes = []
    logging.warning("警告：未能在环境变量中找到 'VIP_CODES'。VIP 码列表为空。")

# --- 应用和缓存配置 ---
app = Quart(__name__)
app = cors(app, allow_origin="*")

# 创建可以在所有 API 请求中共享的 solver 实例
captcha_solver = CaptchaSolver()

# --- 缓存和预生成逻辑 ---
MAX_CACHE_SIZE = 30
TICKET_EXPIRATION_SECONDS = 150


ticket_deque = collections.deque(maxlen=MAX_CACHE_SIZE)
deque_lock = asyncio.Lock()


async def pre_generate_tickets():
    """
    后台任务：持续清理过期Ticket并生成新的Ticket放入缓存池
    """
    logging.info("启动 Ticket 预生成和清理后台任务...")
    while True:
        try:
            async with deque_lock:
                now = time.time()
                # 只要队列不为空，且队头的ticket已过期，就清理掉
                while ticket_deque and (now - ticket_deque[0]['timestamp'] > TICKET_EXPIRATION_SECONDS):
                    ticket_deque.popleft()
                    logging.info("后台任务清理了一个过期的 Ticket。")

            # 清理后，检查队列是否已满
            if len(ticket_deque) < MAX_CACHE_SIZE:
                logging.info(f"缓存未满({len(ticket_deque)}/{MAX_CACHE_SIZE})，后台正在生成新的 Ticket...")
                # 在执行耗时的同步操作前，可以先释放锁（如果需要的话），但这里逻辑简单，可以不释放
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, captcha_solver.solve)

                if result:
                    randstr, ticket = result
                    # 加锁写入
                    async with deque_lock:
                        ticket_deque.append({
                            "randstr": randstr,
                            "ticket": ticket,
                            "timestamp": time.time()
                        })
                    logging.info(f"新的 Ticket 已生成并存入缓存。当前缓存数量: {len(ticket_deque)}")
                else:
                    logging.warning("后台生成 Ticket 失败，将在4秒后重试。")
                    await asyncio.sleep(4)
            else:
                logging.info(f"缓存池已满 ({len(ticket_deque)})，暂停生成。")
                await asyncio.sleep(10)

        except Exception as e:
            logging.error(f"预生成任务出现异常: {e}", exc_info=True)
            await asyncio.sleep(8)


#@app.before_serving
#async def startup():
    #asyncio.create_task(pre_generate_tickets())


# --- API Endpoints ---
@app.route("/solve_captcha", methods=['POST'])
async def solve_captcha_endpoint():
    try:
        data = await request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "msg": "请携带接口权限码再运行。"}), 400

        vCode = data.get('vCode')
        if not vCode or vCode not in vipCodes:
            if not vipCodes:
                return jsonify({"status": "error", "msg": "服务端未配置授权码，拒绝所有访问。"}), 400
            return jsonify({"status": "error", "msg": "无效的接口授权码，访问被拒绝。"}), 400

        logging.info("开始处理 TX 打码请求...")

        valid_item = None
        async with deque_lock:
            while ticket_deque:
                cached_item = ticket_deque.popleft()
                if time.time() - cached_item['timestamp'] < TICKET_EXPIRATION_SECONDS:
                    valid_item = cached_item
                    break
                else:
                    logging.warning("API请求时取出一个已过期的 Ticket，丢弃。")

        if valid_item:
            logging.info("成功从缓存中获取有效的 Ticket。")
            return jsonify({
                "status": "success",
                "msg": "获取到来自缓存的验证码！",
                "randstr": valid_item['randstr'],
                "ticket": valid_item['ticket']
            })

        # 如果缓存为空或所有缓存都已过期，则实时生成
        logging.info("缓存为空或无效，执行实时生成...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, captcha_solver.solve)

        if result:
            randstr, ticket = result
            logging.info(f"实时验证码成功解决。Randstr: {randstr[:5]}...")
            return jsonify({
                "status": "success",
                "msg": "解析验证码成功！",
                "randstr": randstr,
                "ticket": ticket
            })
        else:
            logging.error("所有尝试均失败，无法解决验证码。")
            return jsonify({
                "status": "warning",
                "msg": "所有尝试均失败，无法解决验证码。请重新运行尝试获取",
                "randstr": None,
                "ticket": None
            }), 500

    except Exception as e:
        logging.error(f"处理请求时发生未知错误: {e}", exc_info=True)
        return jsonify({
            "status": "warning",
            "msg": "处理请求时发生未知错误。请重新运行尝试获取",
            "randstr": None,
            "ticket": None
        }), 500


@app.route("/", methods=['GET'])
async def read_root():
    return jsonify({"message": "TX Captcha API 正在运行。"})


if __name__ == '__main__':
    config = Config()
    config.bind = ["0.0.0.0:8000"]

    print("在 http://0.0.0.0:8000 上启动服务器")
    if not vipCodes:
        print("提醒：环境变量 'VIP_CODES' 未设置，API 授权将不起作用。")

    asyncio.run(serve(app, config))