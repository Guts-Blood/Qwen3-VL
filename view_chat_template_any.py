"""
查看数据应用 chat template 的结果（支持原始和转换后的数据）
"""
import pandas as pd
import ast
from transformers import AutoTokenizer
from convert_browser_use_format import convert_messages_format
import argparse

csv_path = 'C:\\source\\repos\\SFT_repo\\Qwen3-VL\\claude_df_converted.csv'
#csv_path = 'C:\\source\\repos\\browser-use-data\\o3_claude_4_data.csv'
def view_chat_template(csv_path, num_samples=3, sample_indices=None, 
                      converted=True, save_to_file=None):
    """
    查看数据应用 chat template 的结果
    
    Args:
        csv_path: CSV 文件路径（原始或转换后的）
        num_samples: 要查看的样本数量
        sample_indices: 指定要查看的样本索引列表
        converted: 如果为 True，假设数据已经转换；如果为 False，会自动转换
        save_to_file: 如果提供，将结果保存到文件
    """
    print("=" * 80)
    print(f"加载数据: {csv_path}")
    print(f"数据状态: {'已转换' if converted else '原始（将自动转换）'}")
    print("=" * 80)
    
    # 加载数据
    try:
        df = pd.read_csv(csv_path)
        print(f"总样本数: {len(df)}\n")
    except Exception as e:
        print(f"[错误] 无法加载 CSV 文件: {e}")
        return
    
    # 加载 tokenizer
    print("加载 Qwen3-VL-30B-A3B-Instruct tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", 
            trust_remote_code=True
        )
        print("[OK] Tokenizer 加载成功\n")
    except Exception as e:
        print(f"[错误] 无法加载 tokenizer: {e}")
        print("请确保已安装 transformers: pip install transformers")
        return
    
    # 确定要查看的样本索引
    if sample_indices is None:
        indices = list(range(min(num_samples, len(df))))
    else:
        indices = [i for i in sample_indices if 0 <= i < len(df)]
    
    if not indices:
        print("没有有效的样本索引")
        return
    
    results = []
    
    for idx in indices:
        print("\n" + "=" * 80)
        print(f"样本 {idx + 1}/{len(df)} (索引 {idx})")
        print("=" * 80)
        
        try:
            # 解析 messages
            messages_str = df.iloc[idx]['messages']
            messages = ast.literal_eval(messages_str) if isinstance(messages_str, str) else messages_str
            
            # 如果数据未转换，先转换
            if not converted:
                print("数据未转换，正在转换...")
                messages = convert_messages_format(messages)
                print("[OK] 转换完成")
            
            print(f"\nMessages 结构:")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'N/A')
                content = msg.get('content', '')
                # 显示前100个字符，去掉换行
                content_preview = content[:150].replace('\n', ' ')
                if len(content) > 150:
                    content_preview += "..."
                print(f"  [{i}] {role}: {len(content)} 字符")
                print(f"      预览: {content_preview}")
            
            # 应用 chat template
            print(f"\n应用 chat template...")
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"格式化后文本长度: {len(formatted_text)} 字符")
            
            # 显示结果（显示关键部分）
            print("\n" + "-" * 80)
            print("应用 Chat Template 后的结果:")
            print("-" * 80)
            
            # 显示开头部分（system prompt）
            if '<|im_start|>system' in formatted_text:
                system_end = formatted_text.find('<|im_start|>user')
                if system_end > 0:
                    print("System Prompt 部分:")
                    print(formatted_text[:system_end])
                    print("...")
            
            # 显示 user 部分
            if '<|im_start|>user' in formatted_text:
                user_start = formatted_text.find('<|im_start|>user')
                user_end = formatted_text.find('<|im_start|>assistant', user_start)
                if user_end > user_start:
                    print("\nUser Message 部分:")
                    user_part = formatted_text[user_start:user_end]
                    print(user_part[:500])
                    if len(user_part) > 500:
                        print("...")
            
            # 显示 assistant 部分（这是最重要的）
            if '<|im_start|>assistant' in formatted_text:
                assistant_start = formatted_text.find('<|im_start|>assistant')
                print("\nAssistant 输出部分（完整）:")
                assistant_part = formatted_text[assistant_start:]
                print(assistant_part)
            
            print("-" * 80)
            
            results.append({
                'index': idx,
                'messages': messages,
                'formatted_text': formatted_text,
                'length': len(formatted_text)
            })
            
        except Exception as e:
            print(f"[错误] 处理样本 {idx + 1} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果指定了保存文件，保存结果
    if save_to_file:
        print(f"\n保存结果到: {save_to_file}")
        with open(save_to_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write("=" * 80 + "\n")
                f.write(f"样本 {result['index'] + 1} (索引 {result['index']})\n")
                f.write("=" * 80 + "\n")
                f.write(f"文本长度: {result['length']} 字符\n")
                f.write("-" * 80 + "\n")
                f.write(result['formatted_text'])
                f.write("\n" + "=" * 80 + "\n\n")
        print("[OK] 结果已保存")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    if results:
        lengths = [r['length'] for r in results]
        print(f"查看的样本数: {len(results)}")
        print(f"文本长度统计:")
        print(f"  最小: {min(lengths)} 字符")
        print(f"  最大: {max(lengths)} 字符")
        print(f"  平均: {sum(lengths) / len(lengths):.1f} 字符")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看数据应用 chat template 的结果')
    parser.add_argument('--input', type=str, default='claude_df.csv',
                       help='CSV 文件路径（默认: claude_df.csv）')
    parser.add_argument('-n', '--num-samples', type=int, default=3,
                       help='要查看的样本数量（默认: 3）')
    parser.add_argument('-i', '--indices', type=int, nargs='+',
                       help='指定要查看的样本索引（从0开始），例如: -i 0 5 9')
    parser.add_argument('--converted', action='store_true',
                       help='标记数据已经转换（如果数据已经转换过，使用此选项）')
    parser.add_argument('--save', type=str, default=None,
                       help='将结果保存到文件（可选）')
    
    args = parser.parse_args()
    
    view_chat_template(
        csv_path=args.input,
        num_samples=args.num_samples,
        sample_indices=args.indices,
        converted=args.converted,
        save_to_file=args.save
    )

