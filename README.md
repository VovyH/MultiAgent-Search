# MultiAgent-Search：基于多智能体识别图像位置

## 0.鸣谢
感谢 [**上海人工智能实验室书生浦语大模型**](https://internlm.openxlab.org.cn/) 以及 [**熊玉洁导师**](https://xiongyujie.cn) 的支持，感谢实训营提供的赞助和机会。
<div align="center">
     <img src="https://github.com/user-attachments/assets/ef94a720-b59a-4dae-b915-8a78be6d2680" alt="Search-SongJiang" />
</div>

## 1.展示(点击进入)：

<div align="center">
     <a href="https://www.bilibili.com/video/BV1BcUmYtESj/?spm_id_from=333.1387.homepage.video_card.click" target="_blank">
          <img width="70%" src="https://github.com/user-attachments/assets/809616ce-0e28-44e6-b9ac-1cadae297234" alt="bilibili" />
     </a>
</div>

## 快速运行：
1. 创建 `.env` 文件（参考 `.env.example`）并配置以下 API 密钥：
   - `DASHSCOPE_API_KEY`：通义千问VL大模型API密钥
   - `BOCHA_API_KEY`：博查搜索API密钥
   - `AMAP_API_KEY`：高德地图API密钥
   - `SILICON_FLOW_API_KEY`：Silicon Flow API密钥（用于向量嵌入）
2. 安装依赖：`pip install -r requirements.txt`
3. 运行服务：`python center/center.py`

## 2.介绍
**MultiAgent-Search**是基于InternLM书生浦语大模型实现的多智能体项目，在图寻地址方面为全球首创，旨在利用多**Agent**识别上海市松江大学城图像，识别过程很好地模拟了人类思考图像位置的过程，以进行图像寻址，如：上海工程技术大学松江校区-图书馆等，在图像寻址功能上超越了**ChatGPT4o**与**文心一言3.5**等大模型。

<div align="center">
    <img src="https://github.com/user-attachments/assets/bcd6ee2e-adcc-4e2e-8461-f79f0ca50303" alt="Search-SongJiang" />
</div>

## 3.功能展示图
![61d7550e63e6facb77c8407de8ade47](https://github.com/user-attachments/assets/4483848c-5074-4cc9-9d05-30c828484585)


## 4.效果对比
### 4.1. ChatGPT4o在图像寻址上的效果，如下图识别上海工程技术大学松江校区-图书馆：
<div align="center">
    <img src="https://github.com/user-attachments/assets/1faaf813-0759-47d7-a070-11b44ede3053" alt="ChatGPT4o" />
</div>

**结论：** `ChatGPT4o`在图像寻址上效果并不好，没有达到人类思考图像位置的能力。

### 4.2. 文心一言3.5在图像寻址上的效果，如下图识别上海工程技术大学松江校区-图书馆：
<div align="center">
    <img src="https://github.com/user-attachments/assets/7f29e7dc-4285-4129-921e-7d00e5fc2fee" alt="文心一言3.5" />
</div>

**结论：** 同理，可以发现`文心一言3.5`在图像寻址上效果并不好，也并没有达到人类思考图像位置的能力。

## 5.最终效果
![903a6230f0c46df636a4a3f3cb1541e](https://github.com/user-attachments/assets/cd43be64-b005-48cb-b07e-31cef6574eff)


## 6. 引用

```bibtex
@misc{Wu2024MultiAgentSearch,
    title={MultiAgent-Search: 基于多智能体识别图像位置},
    author={Yuhang Wu and Henghua Zhang},
    year={2024},
    url={<url id="cuqmhcd43355nsg2o9dg" type="url" status="parsed" title="GitHub - Wuyuhang11/MultiAgent-Search" wc="6723">https://github.com/Wuyuhang11/MultiAgent-Search</url>},
}
```

## 项目简介

MultiAgent-Search是一个基于图像特征提取和相似度计算的搜索系统，用于比较用户上传的图像与其他图像的相似性，并按相似度排序返回结果。

## 系统要求

- Python 3.7+
- pip 包管理工具

## 安装与配置

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/MultiAgent-Search.git
cd MultiAgent-Search
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果`requirements.txt`不存在，可以安装以下基本依赖：

```bash
pip install flask flask-cors Pillow requests numpy scikit-learn
```

### 3. 配置API密钥

本项目需要配置以下API密钥：

1. 创建`.env`文件（或将现有的`.env.example`复制为`.env`）
2. 添加以下内容：

```