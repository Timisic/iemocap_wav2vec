import os
import shutil
import pandas as pd

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置音频源文件夹和目标文件夹路径
AUDIO_SOURCE_DIR = os.path.join(BASE_DIR, 'src_competency' ,'audio')
AUDIO_TARGET_DIR = os.path.join(BASE_DIR, 'src_competency', 'matched_audio_2')

# 设置标注文件路径
LABEL_FILE = os.path.join(BASE_DIR, 'src_competency', 'labels_2.xlsx')

def copy_matched_audio():
    # 创建目标文件夹
    os.makedirs(AUDIO_TARGET_DIR, exist_ok=True)
    
    # 读取标注文件，指定 sheet 名称
    df = pd.read_excel(LABEL_FILE, sheet_name='Sheet1')
    label_ids = set(df['id'].astype(str))
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(AUDIO_SOURCE_DIR) if f.endswith('.wav')]
    
    # 复制匹配的音频文件
    matched_count = 0
    for audio_file in audio_files:
        file_name = os.path.splitext(audio_file)[0]
        if file_name in label_ids:
            src_path = os.path.join(AUDIO_SOURCE_DIR, audio_file)
            dst_path = os.path.join(AUDIO_TARGET_DIR, audio_file)
            shutil.copy2(src_path, dst_path)
            matched_count += 1
    
    print(f'总共找到 {len(audio_files)} 个音频文件')
    print(f'标注文件中有 {len(label_ids)} 个ID')
    print(f'成功匹配并复制了 {matched_count} 个音频文件')

if __name__ == '__main__':
    copy_matched_audio()
