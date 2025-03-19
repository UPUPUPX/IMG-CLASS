import os
# 设置OpenMP环境变量以避免重复初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import torchvision.models as models
import time
# 导入中文字体支持
from matplotlib.font_manager import FontProperties

# 设置torch缓存目录为本地目录
os.environ['TORCH_HOME'] = os.path.join('D:\\CODE\\article\\homework', 'torch_cache')
# 解决torch.classes.__path__._path问题
torch.classes.__path__ = []

# 设置中文字体 - 使用系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面配置
st.set_page_config(
    page_title="图像分类应用",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载PyTorch模型 - 使用轻量级模型
@st.cache_resource
def load_pytorch_model(model_name):
    # 导入权重类
    from torchvision.models import ResNet18_Weights, VGG11_Weights, DenseNet121_Weights
    from torchvision.models import MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
    from torchvision.models import ShuffleNet_V2_X0_5_Weights, MNASNet0_5_Weights
    
    # 使用更轻量级的模型和权重映射
    model_dict = {
        "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "vgg11": (models.vgg11, VGG11_Weights.IMAGENET1K_V1),
        "densenet121": (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1),
        "mobilenet_v3_small": (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
        "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
        "mnasnet0_5": (models.mnasnet0_5, MNASNet0_5_Weights.IMAGENET1K_V1)
    }
    
    # 使用权重枚举而不是pretrained=True
    model_fn, weights = model_dict[model_name]
    model = model_fn(weights=weights)
    model.eval()
    
    # 获取类别信息
    model.idx_to_class = weights.meta["categories"]
    return model

# 检查模型是否已下载到本地
def is_model_downloaded(model_name):
    cache_dir = os.path.join('D:\\CODE\\article\\homework', 'torch_cache', 'hub', 'checkpoints')
    if not os.path.exists(cache_dir):
        return False
    
    patterns = {
        "resnet18": "resnet18",
        "vgg11": "vgg11",
        "densenet121": "densenet121",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "efficientnet_b0": "efficientnet_b0",
        "shufflenet_v2_x0_5": "shufflenet_v2",
        "mnasnet0_5": "mnasnet0_5"
    }
    
    pattern = patterns.get(model_name, model_name)
    
    for file in os.listdir(cache_dir):
        if pattern in file:
            return True
            
    return False

# 获取已下载的模型列表
def get_downloaded_models():
    model_names = ["resnet18", "vgg11", "densenet121", "mobilenet_v3_small", 
                 "efficientnet_b0", "shufflenet_v2_x0_5", "mnasnet0_5"]
    downloaded = []
    for model_name in model_names:
        if is_model_downloaded(model_name):
            downloaded.append(model_name)
    return downloaded

# 下载模型
def download_model(model_name):
    try:
        with st.spinner(f'正在下载 {model_name} 模型...'):
            model = load_pytorch_model(model_name)
            # 验证模型加载成功
            if model is not None:
                return True
            else:
                st.error(f"下载 {model_name} 模型失败：模型为空")
                return False
    except Exception as e:
        st.error(f"下载 {model_name} 模型失败: {str(e)}")
        return False

# 检查并初始化模型
def check_and_initialize_models():
    # 获取已下载的模型
    downloaded_models = get_downloaded_models()
    
    # 如果没有模型，显示下载界面
    if not downloaded_models:
        st.markdown("<h2 class='subtitle'>首次运行需要下载模型</h2>", unsafe_allow_html=True)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 模型列表
        model_names = ["resnet18", "vgg11", "densenet121", "mobilenet_v3_small", 
                      "efficientnet_b0", "shufflenet_v2_x0_5", "mnasnet0_5"]
        
        # 下载所有模型
        downloaded_count = 0
        for i, model_name in enumerate(model_names):
            status_text.text(f"正在下载模型 {model_name} ({i+1}/{len(model_names)})...")
            success = download_model(model_name)
            
            if success:
                downloaded_count += 1
                st.success(f"{model_name} 下载成功!")
            else:
                st.warning(f"{model_name} 下载未完成，稍后将重试")
            
            # 更新进度条
            progress_bar.progress((i + 1) / len(model_names))
        
        # 检查下载结果
        if downloaded_count > 0:
            status_text.text(f"已成功下载 {downloaded_count}/{len(model_names)} 个模型!")
            time.sleep(1)
            st.rerun()
        else:
            status_text.text("所有模型下载失败，请检查网络连接或权限")
    
    return downloaded_models

# 主应用
def main():
    # 设置标题
    st.markdown("<h1 class='title'>图像分类应用</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 添加AI检测声明
    st.warning("⚠️ AI检测结果仅供参考，检测准确性受图像质量、模型能力等多因素影响，请注意辨别实际情况。")
    
    # 检查并初始化模型
    available_models = check_and_initialize_models()
    
    if not available_models:
        return  # 如果没有可用模型，直接返回
    
    # 侧边栏
    with st.sidebar:
        st.header("分类设置")
        model_options = ["默认投票"] + available_models
        model_name = st.selectbox(
            "选择模型", 
            model_options,
            format_func=lambda x: "多模型投票" if x == "默认投票" else x
        )
        
        st.markdown("---")
        st.write("这是一个图像分类的应用，可以识别图片中的物体")
    
    # 创建一个容器来对齐所有元素
    main_container = st.container()
    
    # 上传图片
    with main_container:
        uploaded_file = st.file_uploader("上传一张图片进行分类", type=["jpg", "jpeg", "png", "bmp", "tiff", "gif"])
    
    if uploaded_file is not None:
        try:
            # 显示原始图片 - 调整为480x480
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            
            # 计算调整后的尺寸，保持宽高比例，但高度不超过480
            width, height = img.size
            new_height = min(480, height)
            new_width = int((new_height / height) * width)
            
            # 调整图片大小
            img = img.resize((new_width, new_height))
            
            with main_container:
                # 显示图片在主容器中以保持一致宽度
                st.image(img, caption="上传的图片", use_container_width=True)
            
                # 执行单模型分类
                if model_name != "默认投票":
                    with st.spinner(f'执行{model_name}图像分类中...'):
                        try:
                            # 加载模型
                            pytorch_model = load_pytorch_model(model_name)
                            
                            # 将图像转换为支持的格式 - 所有轻量级模型都使用224x224
                            img_pytorch = img.convert('RGB').resize((224, 224))
                            x = np.array(img_pytorch).transpose((2, 0, 1)) / 255.0
                            # 标准化
                            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                            x = (x - mean) / std
                            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                            
                            # 运行分类
                            with torch.no_grad():
                                preds = pytorch_model(x)
                            
                            # 获取前5个预测
                            probs = torch.nn.functional.softmax(preds, dim=1)[0]
                            top_probs, top_indices = torch.topk(probs, 5)
                            top_probs = top_probs.numpy()
                            top_indices = top_indices.numpy()
                            
                            # 创建结果数据框
                            classes = [pytorch_model.idx_to_class[idx] for idx in top_indices]
                            probs = [float(prob) for prob in top_probs]
                            
                            # 创建并按置信度降序排列结果
                            results_df = pd.DataFrame({
                                "类别": classes,
                                "置信度": probs
                            })
                            # 已经是按置信度降序的，因为torch.topk返回的结果已经排序
                            
                            # 显示结果 - 在同一容器中显示结果
                            st.success(f"最佳预测结果: {classes[0]} (置信度: {probs[0]:.2f})")
                            
                            # 使用单列布局展示结果
                            with main_container:
                                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                                st.write(f"### {model_name}分类结果")
                                st.markdown('<div class="result-table">', unsafe_allow_html=True)
                                st.dataframe(results_df, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # 绘制概率条形图 - 只显示英文标签和置信度
                            with main_container:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # 简化条形图，仅使用英文标签
                                ax.barh(range(len(classes)), probs, color='lightgreen')
                                ax.set_yticks(range(len(classes)))
                                # 不显示y轴标签
                                ax.set_yticklabels([])
                                ax.set_xlabel('Confidence')
                                ax.set_title(f'{model_name} Classification Results', fontname='Microsoft YaHei')
                                
                                # 添加文本标注，只显示英文类别名和置信度
                                for i, (cls, prob) in enumerate(zip(classes, probs)):
                                    # 仅显示类别名(英文)和置信度
                                    ax.text(prob/2, i, f"{cls}: {prob:.2f}", 
                                            ha='center', va='center', 
                                            color='black', fontsize=9,
                                            fontname='Microsoft YaHei')
                                
                                # 调整图表大小以适应容器
                                plt.tight_layout()
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"{model_name}图像分类错误: {str(e)}")
                            import traceback
                            st.error(f"详细错误信息: {traceback.format_exc()}")
                else:
                    # 默认投票机制 - 使用所有已下载的模型
                    model_names = available_models
                    votes = {}
                    all_model_results = {}
                    # 创建进度条
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    for idx, name in enumerate(model_names):
                        with st.spinner(f'执行{name}图像分类中...'):
                            try:
                                # 更新进度
                                progress_bar.progress((idx + 1) / len(model_names))
                                progress_text.text(f"正在使用 {name} 进行分析... ({idx+1}/{len(model_names)})")
                                # 加载模型
                                pytorch_model = load_pytorch_model(name)
                                img_pytorch = img.convert('RGB').resize((224, 224))
                                x = np.array(img_pytorch).transpose((2, 0, 1)) / 255.0
                                # 标准化
                                mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                                std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                                x = (x - mean) / std
                                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                                with torch.no_grad():
                                    preds = pytorch_model(x)
                                probs = torch.nn.functional.softmax(preds, dim=1)[0]
                                top_prob, top_idx = torch.topk(probs, 1)
                                top_prob = top_prob.item()
                                top_idx = top_idx.item()
                                # 修复: 将函数调用改为列表索引
                                class_name = pytorch_model.idx_to_class[top_idx]
                                all_model_results[name] = (class_name, top_prob)
                                if class_name in votes:
                                    votes[class_name] += top_prob
                                else:
                                    votes[class_name] = top_prob
                            except Exception as e:
                                st.error(f"{name}图像分类错误: {str(e)}")
                                import traceback
                                st.error(f"详细错误信息: {traceback.format_exc()}")
                    progress_text.text("分析完成!")
                    
                    # 创建投票结果的可视化
                    if votes:
                        # 对投票结果进行排序
                        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                        
                        # 创建结果数据框
                        vote_classes = [item[0] for item in sorted_votes]
                        vote_scores = [item[1] for item in sorted_votes]
                        
                        # 获取最终预测类别（得票最高的类别）
                        final_class = vote_classes[0] if vote_classes else "未知"
                        
                        # 显示最终投票结果
                        st.success(f"最佳预测结果(投票): {final_class} (累计置信度: {vote_scores[0]:.2f})")
                        
                        # 使用单列布局展示结果
                        with main_container:
                            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                            st.write("### 所有模型预测结果")
                            
                            # 创建包含所有模型预测结果的表格
                            model_results = []
                            for model, (class_name, prob) in all_model_results.items():
                                model_results.append({"模型": model, "预测类别": class_name, "置信度": prob})
                            
                            # 将模型结果按置信度降序排列
                            model_df = pd.DataFrame(model_results)
                            model_df = model_df.sort_values(by="置信度", ascending=False)
                            model_df["置信度"] = model_df["置信度"].apply(lambda x: f"{x:.2f}")
                            st.markdown('<div class="result-table">', unsafe_allow_html=True)
                            st.dataframe(model_df, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 绘制投票结果条形图
                        with main_container:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # 简化条形图，仅使用英文标签
                            top_n = min(5, len(vote_classes))  # 只显示前5个
                            ax.barh(range(top_n), vote_scores[:top_n], color='coral')
                            ax.set_yticks(range(top_n))
                            # 不显示y轴标签
                            ax.set_yticklabels([])
                            ax.set_xlabel('Cumulative Confidence')
                            ax.set_title('Model Voting Results', fontname='Microsoft YaHei')
                            
                            # 添加文本标注，只显示英文类别名和置信度
                            for i, (cls, score) in enumerate(zip(vote_classes[:top_n], vote_scores[:top_n])):
                                ax.text(score/2, i, f"{cls}: {score:.2f}", 
                                        ha='center', va='center', 
                                        color='black', fontsize=9,
                                        fontname='Microsoft YaHei')
                            
                            # 调整图表大小以适应容器
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("无法获取投票分类结果，请尝试使用单个模型进行分类")

        except Exception as e:
            st.error(f"图像分类错误: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")

    # 底部信息
    st.markdown("---")
    st.caption("© 2025 云南大学 软件学院 高级软件设计与体系结构课程")

# 运行主应用
if __name__ == "__main__":
    main()