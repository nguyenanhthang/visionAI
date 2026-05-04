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

pip install PySide6 --proxy http://V3076521:F%40xc0nn@10.222.10.46:3128 --timeout 300 --trusted-host pypi.org --trusted-host files.pythonhosted.org