import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Vẽ Confusion Matrix chuyên nghiệp cho các câu hỏi Closed-ended (Yes/No).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=15)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    return plt

def plot_radar_chart(model_names, metrics_data, categories, title='Model Comparison (All Variants)'):
    """
    Vẽ biểu đồ Radar để so sánh 5 biến thể trên nhiều tiêu chí (Accuracy, BLEU, ROUGE, BERTScore).
    metrics_data: List of lists, mỗi list là chỉ số của 1 model.
    """
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model_name in enumerate(model_names):
        values = metrics_data[i]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=20, y=1.1)
    return plt

def plot_training_history(history, title='Training History'):
    """
    Vẽ đồ thị Loss và Accuracy trong quá trình huấn luyện.
    history: dict có keys 'train_loss', 'val_acc', v.v.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_title('Accuracy Evolution')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return plt

def plot_benchmark_comparison(results_df, metric='Accuracy'):
    """
    Biểu đồ cột so sánh một chỉ số cụ thể giữa các mô hình.
    results_df: DataFrame có cột 'Model' và các chỉ số.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Model', y=metric, data=results_df, palette='viridis')
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=11)
                    
    plt.title(f'Comparison of {metric} across Variants', fontsize=15)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    return plt

def plot_accuracy_by_category(data_df, category_col='Organ', title='Accuracy by Medical Category'):
    """
    Biểu đồ cột phân nhóm để so sánh độ chính xác giữa các cơ quan hoặc loại câu hỏi.
    data_df: DataFrame có cột category_col, 'Model', và 'Correct' (bool).
    """
    acc_df = data_df.groupby([category_col, 'Model'])['Correct'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_col, y='Correct', hue='Model', data=acc_df)
    plt.title(title, fontsize=15)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt

def plot_semantic_distribution(model_scores_dict, title='Semantic Score Distribution (LLM-Judge)'):
    """
    Vẽ biểu đồ Violin để so sánh phân bổ điểm số ngữ nghĩa giữa các model (ví dụ B2 vs DPO).
    model_scores_dict: {'Model A': [scores], 'Model B': [scores]}
    """
    data = []
    for model, scores in model_scores_dict.items():
        for s in scores:
            data.append({'Model': model, 'Score': s})
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Model', y='Score', data=df, inner="quart", palette="Set3")
    plt.title(title, fontsize=15)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    return plt

def plot_latency_vs_accuracy(model_stats, title='Accuracy vs. Latency Trade-off'):
    """
    Biểu đồ bong bóng so sánh Tốc độ và Độ chính xác.
    model_stats: List of dicts [{'name': 'A1', 'accuracy': 0.8, 'latency': 0.1, 'params': 100M}, ...]
    """
    df = pd.DataFrame(model_stats)
    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(df['latency'], df['accuracy'], 
                         s=df['params_mb']*10, # Kích thước bong bóng theo số lượng tham số
                         alpha=0.5, c=np.arange(len(df)), cmap='viridis')
    
    for i, txt in enumerate(df['name']):
        plt.annotate(txt, (df['latency'][i], df['accuracy'][i]), fontsize=12)
        
    plt.xlabel('Latency (seconds/sample)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt

def plot_calibration_curve(y_true, y_probs, n_bins=10, title='Calibration Curve (Reliability)'):
    """
    Biểu đồ hiệu chuẩn để xem độ tin cậy của xác suất dự đoán.
    y_true: nhãn thực tế [0, 1]
    y_probs: xác suất dự đoán lớp 1
    """
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, "s-", label='Model')
    plt.plot([0, 1], [0, 1], "k--", label='Perfectly Calibrated')
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_performance_vs_length(questions, corrects, title='Accuracy vs. Question Length'):
    """
    Biểu đồ xem độ chính xác có giảm khi câu hỏi dài hơn không.
    questions: list các câu hỏi.
    corrects: list các giá trị bool (đúng/sai).
    """
    lengths = [len(q.split()) for q in questions]
    df = pd.DataFrame({'Length': lengths, 'Correct': corrects})
    # Chia nhóm độ dài (bins)
    df['Length_Group'] = pd.cut(df['Length'], bins=[0, 5, 10, 15, 20, 30, 50], 
                               labels=['1-5', '6-10', '11-15', '16-20', '21-30', '31+'])
    
    acc_by_len = df.groupby('Length_Group')['Correct'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Length_Group', y='Correct', data=acc_by_len, marker='o', color='red')
    plt.title(title, fontsize=15)
    plt.ylabel('Accuracy')
    plt.xlabel('Question Length (words)')
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    return plt