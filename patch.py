import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

old_code = '''          navBar.appendChild(btn);
        });'''
new_code = '''          if (!navBar.querySelector('.nav-tabs-buttons')) {
            var inner = document.createElement('div');
            inner.className = 'nav-tabs-inner';
            var leftSpacer = document.createElement('div');
            var btns = document.createElement('div');
            btns.className = 'nav-tabs-buttons';
            inner.appendChild(leftSpacer);
            inner.appendChild(btns);
            navBar.appendChild(inner);
          }
          navBar.querySelector('.nav-tabs-buttons').appendChild(btn);
        });'''

html = html.replace(old_code, new_code)
html = html.replace(old_code.replace('\n', '\r\n'), new_code)

with open('frontend/index.html', 'w', encoding='utf-8', newline='') as f:
    f.write(html)
print('Updated index.html')
