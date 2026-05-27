
pip freeze > requirements.txt
pip download -r requirements.txt -d packages
pip install --no-index --find-links=packages -r requirements.txt
pip install --no-index --find-links=packages ultralytics
venv\Scripts\activate  
Solution: Change the PowerShell execution policy

There are several levels of execution policy:
To allow script execution for the current PowerShell session only (safe and recommended):

✅ Option 1: Temporarily allow scripts (recommended)

Run this in PowerShell as Administrator:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


Then try again:

.\tensorvn\Scripts\Activate.ps1

✅ Option 2: Allow scripts permanently (requires admin rights)

If you want to change it permanently (less secure):

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


💡 RemoteSigned allows local scripts to run, and requires downloaded scripts to be signed.

⚠️ Important Notes:

Use .\ instead of / for file paths in PowerShell:

.\tensorvn\Scripts\Activate.ps1


If you're using Command Prompt (cmd) instead of PowerShell, use:

tensorvn\Scripts\activate.bat


pip install PySide6-6.7.1-6.7.1-cp311-cp311-win_amd64.whl --no-deps


//mở cmd quyền adminstrator
python -m venv D:\thang\tensor_enviroment\myenv
D:\thang\tensor_enviroment\myenv\Scripts\activate
cd D:\thang\tensor_enviroment\project_folder\
pip install --no-index --find-links=packages -r requirements.txt


//mở python folder khác
f:\app\python.exe -m venv D:\Folder_python\thang\tensor\tensors

pip install --proxy http://V3076521:F%40xc0nn@10.222.10.46:3128 --timeout 300 --trusted-host pypi.org --trusted-host files.pythonhosted.org PySide6>=6.5.0 opencv-python>=4.8.0 numpy>=1.24.0 Pillow>=10.0.0 PyYAML


C:\Users\Admin>npm config set proxy http://V3076521:F%40xc0nn@10.222.10.46:3128

C:\Users\Admin>npm install --global yarn
PS D:\python\thang\project\test\UI> $proxy = "http://V3076521:F%40xc0nn@10.222.10.46:3128"
PS D:\python\thang\project\test\UI> $env:HTTP_PROXY  = $proxy
PS D:\python\thang\project\test\UI> $env:HTTPS_PROXY = $proxy
PS D:\python\thang\project\test\UI> $env:http_proxy  = $proxy
PS D:\python\thang\project\test\UI> $env:https_proxy = $proxy
PS D:\python\thang\project\test\UI> $env:ELECTRON_GET_USE_PROXY = "1"
PS D:\python\thang\project\test\UI> yarn add electron --dev
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"


git config --global http.proxy http://V3076521:F%40xc0nn@10.222.10.46:3128
git config --global https.proxy http://V3076521:F%40xc0nn@10.222.10.46:3128
pnpm config set proxy http://V3076521:F%40xc0nn@10.222.10.46:3128
pnpm config set https-proxy http://V3076521:F%40xc0nn@10.222.10.46:3128
pnpm config set strict-ssl false





# Set proxy cho git
git config --global http.proxy http://V3076521:F%40xc0nn@10.222.10.46:3128
git config --global https.proxy http://V3076521:F%40xc0nn@10.222.10.46:3128

# Push lại 
git push -u origin main



git config --global http.sslVerify false
git push -u origin main



https://claude.ai/code/session_01CeL2ZrrkQQSQBZRr6t9B2E