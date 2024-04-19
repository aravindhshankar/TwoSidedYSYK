def select_lines_with_word(input_file, output_file, keyword):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        for i, line in enumerate(lines):
            if line.startswith(keyword):
                start_index = max(0, i - 2)
                end_index = min(i + 3, len(lines))
                selected_lines = lines[start_index:end_index]
                f_out.write('\n'.join(selected_lines))

# Example usage:
input_file = '17AprxFiddle.out'  # Change this to the name of your input file
output_file = 'Processed17AprxFiddle.out'  # Change this to the name of your output file
keyword = 'Nbig'  # Change this to the word you want to select lines starting with
select_lines_with_word(input_file, output_file, keyword)
