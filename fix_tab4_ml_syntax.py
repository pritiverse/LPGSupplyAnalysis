import re

with open('lpg_dashboard/tabs/tab4_ml.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the exact syntax error at line 1043
fixed = re.sub(r'if not ok\\(arima\\):\\\\n\\s+empty_state\\("ARIMA forecast checkpoint not found.\\"\\)\\\\n\\s+else:\\n\\s+else:', 
               r'if not ok(arima):\n            empty_state("ARIMA forecast checkpoint not found.")\n        else:', content)

with open('lpg_dashboard/tabs/tab4_ml.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Syntax fixed in tab4_ml.py')
