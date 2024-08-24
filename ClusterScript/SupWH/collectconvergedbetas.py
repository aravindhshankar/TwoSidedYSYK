import re


def ret_converged_betas(filename='OutputForalphazeroSupCondlamb0_05.out'):
    # Step 1: Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Step 2: Initialize the list for beta values
    beta_values = []

    # Step 3: Process the lines in chunks
    # for i in range(3, 25, 5):
    for i in range(3, len(lines)-3, 5):
        try:
            # Extract the beta value
            beta_line = lines[i + 1].strip(' \n')

            beta = re.findall(r"\b\d+\b", beta_line)
            if beta:
                beta = int(beta[0]) #Found all the betas
            
            # Extract the 'end itern' value
            end_line = lines[i + 3].strip(' \n')
            end_itern = int(end_line.split()[-1]) #found end itern
            if end_itern < 15000:
                beta_values.append(beta)

        except (ValueError, AttributeError):
            continue

    # Step 4: Output the result
    return beta_values
