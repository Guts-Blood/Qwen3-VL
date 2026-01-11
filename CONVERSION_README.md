# Browser-Use 数据格式转换工具

这个工具用于将 browser-use 数据格式转换为训练格式。

## 转换内容

### 输出格式转换
- **原始格式**: `thinking`, `evaluation_previous_goal`, `memory`, `next_goal`, `action` (5个字段)
- **转换后格式**: `thinking`, `tool_call` (2个字段)

### 具体变化
1. **evaluation_previous_goal, memory, next_goal** → 合并到 `thinking` 字段中
2. **action** → 转换为 `tool_call` 字段（格式化为函数调用字符串）
3. **System prompt** → 更新 output 格式说明，移除独立字段描述

## 使用方法

### 1. 测试单行转换
```bash
python convert_browser_use_format.py --test
```

### 2. 处理整个CSV文件
```bash
python convert_browser_use_format.py --input claude_df.csv --output claude_df_converted.csv
```

### 3. 查看转换效果
```bash
python data_demo.py
```

## 函数说明

### `convert_messages_format(messages)`
转换单个 messages 列表的格式。

**参数:**
- `messages`: List[Dict] - 原始 messages 列表

**返回:**
- List[Dict] - 转换后的 messages 列表

### `process_csv_to_new_format(csv_path, output_csv_path=None)`
处理整个CSV文件。

**参数:**
- `csv_path`: str - 输入CSV文件路径
- `output_csv_path`: str (可选) - 输出CSV文件路径，如果为None则返回DataFrame

**返回:**
- DataFrame 或 None（如果指定了输出路径）

## 转换示例

### 原始格式
```json
{
  "thinking": "...",
  "evaluation_previous_goal": "...",
  "memory": "...",
  "next_goal": "...",
  "action": [{"go_to_url": {"url": "https://example.com"}}]
}
```

### 转换后格式
```json
{
  "thinking": "...\n\nEvaluation of Previous Goal: ...\n\nMemory: ...\n\nNext Goal: ...",
  "tool_call": "go_to_url({\"url\": \"https://example.com\", \"new_tab\": false})"
}
```

## 注意事项

- 转换会修改 system prompt 中的 output 格式说明
- tool_call 字段格式化为函数调用字符串，多个调用用换行分隔
- 如果转换过程中出现错误，原始数据会被保留

