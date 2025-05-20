import re
import sys

def process_document(input_path, output_path):
    with open(input_path, 'r', encoding="utf-8") as f:
        text = f.read()
    
    # 按"References"标题将文档分为正文和参考文献部分（仅分割第一次出现）
    parts = re.split(r'(?im)^\s*References\s*$', text, maxsplit=1)
    if len(parts) < 2:
        print("未找到 'References' 部分")
        return
    main_text = parts[0]
    references_block = parts[1]

    # --- 1. 在正文中按顺序提取引用并建立原->新编号映射 ---
    citation_pattern = re.compile(r'\[(\d+)\]')
    mapping = {}
    next_index = 1
    # 利用finditer获取所有匹配的顺序
    for match in citation_pattern.finditer(main_text):
        orig_num = match.group(1)
        if orig_num not in mapping:
            mapping[orig_num] = next_index
            next_index += 1

    # --- 2. 更新正文中的引用序号 ---
    def replace_citation(match):
        orig = match.group(1)
        return f"[{mapping.get(orig, orig)}]"
    new_main_text = citation_pattern.sub(replace_citation, main_text)
    
    # --- 3. 从参考文献部分提取所有引用条目（认为每行一个条目） ---
    # 匹配每行以 [数字] 开头的行
    ref_pattern = re.compile(r'^\s*\[(\d+)\]\s*(.*)$', re.M)
    ref_entries = {}
    for m in ref_pattern.finditer(references_block):
        ref_num = m.group(1)
        content = m.group(2).strip()
        ref_entries[ref_num] = content

    # --- 4. 根据正文中引用顺序生成新的参考文献列表 ---
    # 构造新引用列表，新列表中条目的顺序为 mapping中新编号的升序
    # 为此先反转 mapping： {new_index: orig_index}
    inv_mapping = {new: orig for orig, new in mapping.items()}
    new_refs_lines = []
    for i in range(1, next_index):
        orig_num = inv_mapping.get(i)
        if orig_num and orig_num in ref_entries:
            new_line = f"[{i}] {ref_entries[orig_num]}"
            new_refs_lines.append(new_line)
        else:
            # 如果对应条目不存在，则给个提示（也可选择跳过）
            new_line = f"[{i}] (未找到原参考文献条目)"
            new_refs_lines.append(new_line)
    
    new_references_block = "\n".join(new_refs_lines)

    # --- 5. 组合更新后的正文和参考文献部分 ---
    new_text = new_main_text.rstrip() + "\n\nReferences\n\n" + new_references_block + "\n"
    
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(new_text)

if __name__ == "__main__":
    path = "review/graph signal sampling and reconstruction_original.md"
    process_document(path, "review/graph signal sampling and reconstruction_processed.md")