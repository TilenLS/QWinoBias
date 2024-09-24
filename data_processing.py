import os, sys, glob, re
import pandas as pd

#file = open(os.getcwd() + '/wino/data/male_occupations.txt', 'r')
#lines = file.readlines()
#OCCUPATIONS = [line[:-1] for line in lines[:-1]]
#OCCUPATIONS.append(lines[-1])
#file.close()

OCCUPATIONS = ['driver', 'supervisor', 'janitor', 'cook', 'mover', 
               'laborer', 'construction worker', 'chief', 'developer', 
               'carpenter', 'manager', 'lawyer', 'farmer', 'salesperson', 
               'physician', 'guard', 'analyst', 'mechanic', 'sheriff', 'CEO', # Male Stereotyped Occupations
               'attendant', 'cashier', 'teacher', 'nurse', 'assistant', 
               'secretary', 'auditor', 'cleaner', 'receptionist', 'clerk', 
               'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 
               'baker', 'accountant', 'editor', 'librarian', 'tailor'] # Female Stereotyped Occupations

#path = sys.argv[1]

def get_lines(path: str) -> [str]:
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    return lines

def process_line(line: str) -> tuple[str, str, str, str, str]:
    refs = re.findall('|'.join(OCCUPATIONS), line)
    target_words = re.findall(r'\[.*?\]', line)
    right_ref = target_words[0][1:-1]
    right_ref = re.findall('|'.join(refs), right_ref).pop()
    refs.remove(right_ref)
    wrong_ref = refs.pop()
    pro1 = target_words[1][1:-1]
    pro2 = ''
    if len(target_words) > 2:
        pro2 = target_words[2][1:-1]
    line = re.sub(r'[0-9]|[0-9] |\n|\[|\]', '', line)
    return (line, right_ref, wrong_ref, pro1, pro2)

def to_df(lines: [str]) -> pd.DataFrame:
    data = []
    for line in lines:
        data.append(process_line(line))
    df = pd.DataFrame(data, columns=['Sentence', 'Right Referent', 'Wrong Referent', 'Pronoun 1', 'Pronoun 2'])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == '__main__':
    os.chdir(os.getcwd() + '/wino/data')
    paths = glob.glob("*.test")
    lines = []
    for path in paths:
        lines += get_lines(path)
    df = to_df(lines)
    print(df)
    df.to_pickle('test_data.pkl')
