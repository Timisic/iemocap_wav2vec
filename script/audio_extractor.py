import os
import subprocess


def extract_audio(input_path, output_path):
    """
    从视频文件中提取音频
    """
    try:
        # 确保路径是绝对路径
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：输入文件不存在：{input_path}")
            return

        # 构建ffmpeg命令，移除路径周围的引号
        command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-y',
            output_path
        ]

        # 执行命令时不需要join，直接传递列表
        result = subprocess.run(
            command,  # 直接使用命令列表
            shell=False,  # 改为False，因为我们使用列表形式
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print(f"音频提取成功：{output_path}")
        else:
            print(f"处理文件时出错：\n{result.stderr}")

    except Exception as e:
        print(f"发生错误：{str(e)}")
        print(f"当前处理的文件路径：\n输入：{input_path}\n输出：{output_path}")


def main():
    # 使用原始字符串表示路径
    input_dir = '../src_competency'
    output_dir = '../src_competency/audio'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在：{input_dir}")
        return

    # 检查ffmpeg是否已安装
    try:
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=True)
    except Exception as e:
        print(f"错误：ffmpeg检查失败 - {str(e)}")
        print("请确保ffmpeg已安装并添加到系统环境变量PATH中。")
        return

    # 处理输入目录中的所有视频文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mov', '.mts')):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(output_dir, output_filename)

            print(f"\n处理文件：{filename}")
            print(f"输入路径：{input_path}")
            print(f"输出路径：{output_path}")
            extract_audio(input_path, output_path)


if __name__ == "__main__":
    main()
