import re

string  = 'L ww $$ $$111> $> $1> 114514 but $1919810>'
match = re.findall(r'\$\d*>', string)

print(match)
