import pandas as pd
import ast
import json
import os
from typing import List, Dict, Any
from transformers import AutoProcessor


def format_tool_descriptions(tool_descriptions_path: str = "tool_descriptions_complete.json") -> str:
    """
    从JSON文件中读取工具描述并格式化为文本格式
    
    Args:
        tool_descriptions_path: 工具描述JSON文件路径
    
    Returns:
        格式化的工具描述文本
    """
    try:
        # 尝试从当前目录或脚本所在目录读取文件
        if os.path.exists(tool_descriptions_path):
            json_path = tool_descriptions_path
        else:
            # 尝试从脚本所在目录读取
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, tool_descriptions_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            tool_data = json.load(f)
        
        tools = tool_data.get('tools', [])
        
        if not tools:
            return ""
        
        # 格式化工具描述为system prompt的一部分
        tool_description_lines = []
        tool_description_lines.append("Tool Description:")
        tool_description_lines.append("")
        tool_description_lines.append("You have access to the following tools. Use them in the <tool_call> section of your response.")
        tool_description_lines.append("")
        
        for tool in tools:
            tool_name = tool.get('name', '')
            if not tool_name:
                continue
            
            tool_desc = tool.get('description', '')
            params = tool.get('params', {})
            
            # 工具名称
            tool_description_lines.append(f"- {tool_name}()")
            
            # 工具描述
            if tool_desc:
                tool_description_lines.append(f"  Description: {tool_desc}")
            
            # 参数说明
            if params:
                tool_description_lines.append("  Parameters:")
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', '')
                    param_default = param_info.get('default', None)
                    param_min = param_info.get('minimum') or param_info.get('minLength')
                    param_max = param_info.get('maximum') or param_info.get('maxLength')
                    
                    param_line = f"    * {param_name} ({param_type})"
                    parts = []
                    if param_desc:
                        parts.append(param_desc)
                    if param_default is not None:
                        parts.append(f"default: {param_default}")
                    if param_min is not None and param_max is not None:
                        parts.append(f"range: [{param_min}, {param_max}]")
                    elif param_min is not None:
                        parts.append(f"min: {param_min}")
                    elif param_max is not None:
                        parts.append(f"max: {param_max}")
                    
                    if parts:
                        param_line += ": " + ", ".join(parts)
                    
                    tool_description_lines.append(param_line)
            else:
                # 没有参数的工具
                tool_description_lines.append("  Parameters: None")
            
            tool_description_lines.append("")
        
        return "\n".join(tool_description_lines).rstrip()
    
    except FileNotFoundError:
        print(f"Warning: Tool descriptions file not found: {tool_descriptions_path}")
        return ""
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse tool descriptions JSON: {e}")
        return ""
    except Exception as e:
        print(f"Warning: Error loading tool descriptions: {e}")
        return ""


def convert_assistant_output(assistant_content: str) -> str:
    """
    将assistant的JSON输出转换为标记格式
    格式: <thinking>...</thinking>\n<tool_call>...</tool_call>
    """
    try:
        parsed = json.loads(assistant_content)
        
        # 合并所有信息到thinking
        thinking_parts = []
        
        if 'thinking' in parsed:
            thinking_parts.append(parsed['thinking'])
        
        if 'evaluation_previous_goal' in parsed:
            thinking_parts.append(f"\n\nEvaluation of Previous Goal: {parsed['evaluation_previous_goal']}")
        
        if 'memory' in parsed:
            thinking_parts.append(f"\n\nMemory: {parsed['memory']}")
        
        if 'next_goal' in parsed:
            thinking_parts.append(f"\n\nNext Goal: {parsed['next_goal']}")
        
        # 构建新的thinking
        new_thinking = "".join(thinking_parts)
        
        # 将action转换为tool_call格式
        tool_call = ""
        if 'action' in parsed:
            actions = parsed['action']
            if isinstance(actions, list) and len(actions) > 0:
                # 将action列表转换为tool_call格式的文本描述
                # 格式：function_name({"param1": "value1", "param2": "value2"})
                tool_calls = []
                for action in actions:
                    if isinstance(action, dict):
                        for action_name, action_params in action.items():
                            # 确保参数是有效的JSON格式
                            if isinstance(action_params, dict):
                                params_str = json.dumps(action_params, ensure_ascii=False)
                            else:
                                params_str = json.dumps(action_params, ensure_ascii=False)
                            tool_calls.append(f"{action_name}({params_str})")
                
                tool_call = "\n".join(tool_calls) if tool_calls else ""
        
        # 格式化为标记格式
        result_parts = []
        if new_thinking:
            result_parts.append(f"<thinking>\n{new_thinking}\n</thinking>")
        if tool_call:
            result_parts.append(f"<tool_call>\n{tool_call}\n</tool_call>")
        
        return "\n\n".join(result_parts) if result_parts else ""
    
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse assistant content as JSON: {e}")
        return f"<thinking>\n{assistant_content}\n</thinking>"
    except Exception as e:
        print(f"Warning: Error processing assistant content: {e}")
        return f"<thinking>\n{assistant_content}\n</thinking>"


def convert_user_input(user_content: str, assistant_content: str) -> str:
    """
    转换user输入，将evaluation/memory/next_goal的描述合并到prompt中
    并将action的描述改为tool_call
    """
    # 解析assistant的原始输出，提取信息用于修改prompt
    try:
        parsed = json.loads(assistant_content)
        
        # 如果prompt中包含evaluation_previous_goal, memory, next_goal的描述，需要调整
        # 这里假设原始prompt在user_content中，我们可能需要修改system prompt
        # 但根据实际情况，可能主要修改的是output格式描述
        
        # 替换action为tool_call
        modified_content = user_content.replace('"action":', '"tool_call":')
        modified_content = modified_content.replace("'action':", "'tool_call':")
        modified_content = modified_content.replace("action field", "tool_call field")
        modified_content = modified_content.replace("action list", "tool_call list")
        
        # 将output格式描述中的evaluation/memory/next_goal移到thinking中
        # 这需要更精细的文本处理，这里提供基本框架
        
        return modified_content
    
    except Exception as e:
        print(f"Warning: Error processing user input: {e}")
        return user_content


def convert_messages_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换整个messages列表的格式
    
    输入格式：
    - system: 包含prompt说明
    - user: 包含当前状态和任务
    - assistant: JSON格式输出，包含thinking, evaluation_previous_goal, memory, next_goal, action
    
    输出格式：
    - system: 调整后的prompt（evaluation/memory/next_goal合并到thinking，action改为tool_call）
    - user: 保持不变或微调
    - assistant: 只包含thinking和tool_call
    """
    converted_messages = []
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            # 修改system prompt中的输出格式描述
            new_content = content
            import re
            
            # 移除instruction和增量信息部分（这些应该训练到模型里，而不是作为prompt）
            # 这些都是通用的instruction，我们要训练专门的browser-use agent，所以移除
            sections_to_remove = [
                r'<reasoning_rules>.*?</reasoning_rules>',
                r'<efficiency_guidelines>.*?</efficiency_guidelines>',
                r'<browser_rules>.*?</browser_rules>',
                r'<task_completion_rules>.*?</task_completion_rules>',
                r'<action_rules>.*?</action_rules>',
                r'<examples>.*?</examples>',
                # browser_state 和 browser_vision 作为说明出现在 input 部分，也需要移除
                r'<browser_state>.*?</browser_state>',
                r'<browser_vision>.*?</browser_vision>',
            ]
            
            for pattern in sections_to_remove:
                new_content = re.sub(pattern, '', new_content, flags=re.DOTALL | re.IGNORECASE)
            
            # 清理多余的空行（连续3个或更多空行替换为2个）
            new_content = re.sub(r'\n{3,}', '\n\n', new_content)
            
            # 修复移除sections后留下的不完整编号列表项（如 "3. " 后面没有内容）
            # 匹配单独的编号项（数字后跟点，然后只有空白或换行）
            new_content = re.sub(r'^\s*\d+\.\s*$', '', new_content, flags=re.MULTILINE)
            
            # 重新清理多余的空行
            new_content = re.sub(r'\n{3,}', '\n\n', new_content)
            
            # 替换action为tool_call（如果还有残留）
            new_content = new_content.replace('"action":', '"tool_call":')
            new_content = new_content.replace("'action':", "'tool_call':")
            new_content = new_content.replace('"action"', '"tool_call"')
            new_content = new_content.replace("'action'", "'tool_call'")
            new_content = new_content.replace('action field', 'tool_call field')
            new_content = new_content.replace('action list', 'tool_call list')
            new_content = new_content.replace('action parameter', 'tool_call parameter')
            
            # 修改output格式描述，将evaluation/memory/next_goal合并到thinking
            # 找到output格式部分并修改
            import re
            
            # 替换JSON格式描述
            output_pattern = r'\{[^}]*"thinking"[^}]*"evaluation_previous_goal"[^}]*"memory"[^}]*"next_goal"[^}]*"action"[^}]*\}'
            
            # 更通用的方法：替换整个output section
            if '"evaluation_previous_goal"' in new_content:
                # 说明evaluation应该在thinking中体现
                new_content = new_content.replace(
                    '"evaluation_previous_goal": "One-sentence analysis of your last action. Clearly state success, failure, or uncertain.",',
                    ''
                ).replace(
                    'evaluation_previous_goal": "One-sentence analysis of your last action. Clearly state success, failure, or uncertain.",',
                    ''
                )
                # 在thinking描述中添加evaluation说明
                thinking_desc = 'Your thinking should include: evaluation of previous actions, memory of progress, and next goals.'
                if '"thinking":' in new_content or "'thinking':" in new_content:
                    # 在thinking描述后添加
                    pass  # 已经在thinking中体现了
            
            if '"memory"' in new_content:
                new_content = new_content.replace(
                    '"memory": "1-3 sentences of specific memory of this step and overall progress. You should put here everything that will help you track progress in future steps. Like counting pages visited, items found, etc.",',
                    ''
                )
            
            if '"next_goal"' in new_content:
                new_content = new_content.replace(
                    '"next_goal": "State the next immediate goals and actions to achieve it, in one clear sentence."',
                    ''
                )
            
            # 修改output格式为只有thinking和tool_call
            output_example_pattern = r'\{\s*"thinking":\s*"[^"]*",\s*"evaluation_previous_goal":\s*"[^"]*",\s*"memory":\s*"[^"]*",\s*"next_goal":\s*"[^"]*"\s*"action":\s*\[[^\]]*\]\s*\}'
            
            # 简化output格式描述
            new_output_format = '''{
  "thinking": "Your comprehensive reasoning that includes: evaluation of previous actions, memory of progress, next goals, and analysis of current state.",
  "tool_call": "The tool calls you want to make, formatted as function calls, one per line. Example: go_to_url({"url": "https://example.com", "new_tab": false})"
}'''
            
            # 替换output section - 改为标记格式而非JSON
            import re
            
            # 找到并替换output格式描述
            if '<output>' in new_content and '</output>' in new_content:
                # 先移除所有旧的JSON格式相关描述
                old_patterns = [
                    r'"evaluation_previous_goal":\s*"[^"]*",?\s*\n',
                    r'"memory":\s*"[^"]*",?\s*\n',
                    r'"next_goal":\s*"[^"]*",?\s*\n',
                    r'"action":\s*\[[^\]]*\],?\s*\n',
                    r'"action":\s*"[^"]*",?\s*\n',
                    r'You must ALWAYS respond with a valid JSON[^\n]*\n',
                    r'Action list should NEVER be empty[^\n]*\n',
                ]
                for pattern in old_patterns:
                    new_content = re.sub(pattern, '', new_content, flags=re.IGNORECASE)
                
                # 移除所有JSON示例
                json_example_pattern = r'\{[^}]*"thinking"[^}]*"evaluation_previous_goal"[^}]*\}'
                new_content = re.sub(json_example_pattern, '', new_content)
                json_example_pattern2 = r'\{[^}]*"thinking"[^}]*"memory"[^}]*\}'
                new_content = re.sub(json_example_pattern2, '', new_content)
                json_example_pattern3 = r'\{[^}]*"thinking"[^}]*"next_goal"[^}]*\}'
                new_content = re.sub(json_example_pattern3, '', new_content)
                json_example_pattern4 = r'\{[^}]*"thinking"[^}]*"tool_call"[^}]*\}'
                new_content = re.sub(json_example_pattern4, '', new_content)
                
                # 更新output格式说明为标记格式
                output_section_pattern = r'<output>.*?</output>'
                new_output_section = '''<output>
You must ALWAYS respond in the following format using XML-like tags:

<thinking>
Your comprehensive reasoning that includes:
- Evaluation of previous actions (success/failure/uncertainty)
- Memory of overall progress and specific details that help track progress
- Next goals and immediate actions to achieve them
- Analysis of current state and relevant context
</thinking>

<tool_call>
The tool calls you want to make, formatted as function calls, one per line.
Example: go_to_url({"url": "https://example.com", "new_tab": false})
If multiple tool calls are needed, put each on a separate line.
</tool_call>

Both sections are required. The tool_call section should NEVER be empty.
</output>'''
                new_content = re.sub(output_section_pattern, new_output_section, new_content, flags=re.DOTALL)
            
            # 注意：examples 和 efficiency_guidelines 已经在开头被移除了
            # 这里只需要处理output格式的更新
            
            # 移除output格式说明中提到的JSON相关文字
            new_content = new_content.replace('valid JSON', 'the format')
            new_content = new_content.replace('JSON format', 'specified format')
            new_content = new_content.replace('in JSON', 'in the format')
            
            # 替换所有独立出现的 "action" 引用为 "tool_call"（但要小心不要替换单词的一部分）
            # 只在特定上下文中替换
            new_content = re.sub(r'\baction\b(?=\s*(?:field|list|parameter|name))', 'tool_call', new_content, flags=re.IGNORECASE)
            
            # 在system prompt的最后添加工具描述部分
            tool_descriptions = format_tool_descriptions()
            if tool_descriptions:
                # 确保内容末尾有适当的换行
                new_content = new_content.rstrip()
                if not new_content.endswith('\n'):
                    new_content += '\n\n'
                else:
                    new_content += '\n'
                new_content += tool_descriptions
            
            converted_messages.append({
                "role": role,
                "content": new_content
            })
        
        elif role == 'user':
            # user消息中也需要移除 browser_state 等增量信息部分
            user_content = content
            if isinstance(user_content, str):
                import re
                # 移除 browser_state 部分（这是增量信息，应该训练到模型里）
                user_content = re.sub(r'<browser_state>.*?</browser_state>', '', user_content, flags=re.DOTALL | re.IGNORECASE)
                # 清理多余的空行
                user_content = re.sub(r'\n{3,}', '\n\n', user_content)
                
                converted_messages.append({
                    "role": role,
                    "content": user_content
                })
            else:
                converted_messages.append(msg)
        
        elif role == 'assistant':
            # 转换assistant输出格式为标记格式
            if isinstance(content, str):
                # convert_assistant_output 现在返回标记格式的字符串
                new_content = convert_assistant_output(content)
                
                converted_messages.append({
                    "role": role,
                    "content": new_content
                })
            else:
                converted_messages.append(msg)
        else:
            converted_messages.append(msg)
    
    return converted_messages


def process_csv_to_new_format(csv_path: str, output_csv_path: str = None, chunksize: int = 100):
    """
    处理整个CSV文件，转换所有行的messages格式
    
    Args:
        csv_path: 输入CSV文件路径
        output_csv_path: 输出CSV文件路径，如果为None则返回DataFrame
        chunksize: 分块大小，用于处理大文件（默认100行）
    """
    print(f"Loading CSV file: {csv_path}")
    
    # 首先尝试读取第一行来获取列名和格式
    try:
        sample_df = pd.read_csv(csv_path, nrows=1)
        columns = sample_df.columns.tolist()
        print(f"CSV columns: {columns}")
    except Exception as e:
        print(f"Error reading CSV sample: {e}")
        raise
    
    # 使用分块读取来处理大文件
    print("Converting messages format in chunks...")
    
    chunk_list = []
    total_processed = 0
    first_chunk = True
    
    try:
        for chunk_df in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            chunk_start = total_processed
            print(f"Processing chunk starting at row {chunk_start} ({len(chunk_df)} rows)...")
            
            converted_messages_list = []
            
            for idx, row in chunk_df.iterrows():
                if total_processed % 100 == 0:
                    print(f"Processing row {total_processed}...")
                
                try:
                    messages_str = row['messages']
                    messages = ast.literal_eval(messages_str) if isinstance(messages_str, str) else messages_str
                    
                    converted_messages = convert_messages_format(messages)
                    converted_messages_list.append(str(converted_messages))  # 转换回字符串格式保存
                    
                except Exception as e:
                    print(f"Error processing row {total_processed} (chunk idx {idx}): {e}")
                    try:
                        converted_messages_list.append(str(row['messages']))  # 保持原样
                    except:
                        converted_messages_list.append("[]")  # 如果还是失败，使用空列表
                
                total_processed += 1
            
            chunk_df['messages'] = converted_messages_list
            
            # 如果是第一块，写入模式为'w'并包含header，否则追加模式
            if output_csv_path:
                chunk_df.to_csv(output_csv_path, mode='w' if first_chunk else 'a', 
                              header=first_chunk, index=False)
                first_chunk = False
            else:
                chunk_list.append(chunk_df)
        
        print(f"Total rows processed: {total_processed}")
        
        if output_csv_path:
            print(f"Converted data saved to: {output_csv_path}")
            return None
        else:
            # 合并所有chunks
            df = pd.concat(chunk_list, ignore_index=True)
            return df
            
    except Exception as e:
        print(f"Error processing CSV: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert browser-use data format')
    parser.add_argument('--input', type=str, default='claude_df.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='claude_df_converted.csv', help='Output CSV file path')
    parser.add_argument('--test', action='store_true', help='Test conversion on first row only')
    parser.add_argument('--apply-template', action='store_true', help='Apply chat template to test result (requires PyTorch)')
    
    args = parser.parse_args()
    
    if args.test:
        # 测试模式：只处理第一行
        print("=" * 80)
        print("Testing conversion on first row")
        print("=" * 80)
        
        df = pd.read_csv(args.input, nrows=1)
        messages_str = df['messages'].iloc[0]
        messages = ast.literal_eval(messages_str)
        
        print("\nOriginal messages structure:")
        for i, msg in enumerate(messages):
            print(f"  [{i}] {msg.get('role')}: {len(str(msg.get('content')))} chars")
        
        converted = convert_messages_format(messages)
        
        print("\nConverted messages structure:")
        for i, msg in enumerate(converted):
            print(f"  [{i}] {msg.get('role')}: {len(str(msg.get('content')))} chars")
            if msg.get('role') == 'assistant':
                content_str = str(msg.get('content', ''))
                print(f"      Content preview: {content_str[:300]}...")
                print(f"      Full content:\n{content_str}")
        
        # 可选：应用chat template查看效果
        if args.apply_template:
            print("\n" + "=" * 80)
            print("Applying chat template to converted messages")
            print("=" * 80)
            
            try:
                model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                
                formatted_text = processor.apply_chat_template(
                    converted,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                print(formatted_text[-3000:])  # 显示最后3000个字符（主要是assistant部分）
                
            except Exception as e:
                print(f"Error applying chat template (PyTorch may not be installed): {e}")
                print("You can skip this step if you only need format conversion.")
    
    else:
        # 处理整个文件
        print(f"Processing entire file: {args.input}")
        result_df = process_csv_to_new_format(args.input, args.output)
        
        if result_df is not None:
            print(f"\nConversion complete! DataFrame with {len(result_df)} rows returned.")
            print("First row converted messages preview:")
            first_messages = ast.literal_eval(result_df['messages'].iloc[0])
            for i, msg in enumerate(first_messages):
                if msg.get('role') == 'assistant':
                    print(f"\nAssistant output:\n{msg.get('content', '')[:500]}...")

