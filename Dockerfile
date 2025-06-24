# 使用官方 Python 基础镜像
FROM python:3.12.2

# 设置工作目录
WORKDIR /code

# 安装 OpenCV 所需的系统级依赖
# 使用非交互式前端可以避免安装过程中的提示，并清理 apt 缓存以减小镜像体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 使用清华大学镜像源安装 Python 依赖，以加快下载速度
RUN pip install --upgrade opencv-python
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制所有应用代码
COPY . .

CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8093"]