@echo off
REM 이미지 파일에 대응하는 텍스트 파일이 있는지 확인
for %%f in (*.jpg *.jpeg *.webp *.png) do (
    if not exist "%%~nf.txt" echo Missing caption for: %%f
)

REM 텍스트 파일에 대응하는 이미지 파일이 있는지 확인
for %%f in (*.txt) do (
    if not exist "%%~nf.jpg" if not exist "%%~nf.jpeg" if not exist "%%~nf.webp" if not exist "%%~nf.png" echo Missing image for: %%f
)

pause