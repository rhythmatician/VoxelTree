@echo off
setlocal enabledelayedexpansion

REM ======================================================================
REM  data-harvester — Build and Install
REM
REM  Builds the DataHarvester Fabric mod and copies the JAR to the
REM  Modrinth "LODiffusion dependencies" profile's mods folder.
REM
REM  Prerequisites:
REM    - Java 21+ on PATH
REM    - Gradle wrapper (will be bootstrapped from LODiffusion if missing)
REM ======================================================================

set "SCRIPT_DIR=%~dp0"
set "MOD_DIR=%SCRIPT_DIR%"
set "LODIFFUSION_DIR=%SCRIPT_DIR%\..\..\LODiffusion"
set "MODRINTH_MODS=%APPDATA%\ModrinthApp\profiles\LODiffusion dependencies\mods"

echo.
echo  ================================================
echo   DataHarvester — Build ^& Install
echo  ================================================

REM --- Step 1: Ensure Gradle wrapper files exist -------------------------
if not exist "%MOD_DIR%gradlew.bat" (
    echo.
    echo  Copying Gradle wrapper from LODiffusion...
    if not exist "%LODIFFUSION_DIR%\gradlew.bat" (
        echo  ERROR: Cannot find LODiffusion\gradlew.bat
        echo  Please copy gradlew.bat, gradlew, and gradle\wrapper\gradle-wrapper.jar
        echo  into this directory.
        exit /b 1
    )
    copy "%LODIFFUSION_DIR%\gradlew.bat" "%MOD_DIR%" >nul
    copy "%LODIFFUSION_DIR%\gradlew" "%MOD_DIR%" >nul
    if not exist "%MOD_DIR%gradle\wrapper" mkdir "%MOD_DIR%gradle\wrapper"
    copy "%LODIFFUSION_DIR%\gradle\wrapper\gradle-wrapper.jar" "%MOD_DIR%gradle\wrapper\" >nul
    echo  Done.
)

REM --- Step 2: Build the mod ---------------------------------------------
echo.
echo  Building data-harvester mod...
pushd "%MOD_DIR%"
call gradlew.bat build
if errorlevel 1 (
    echo.
    echo  ERROR: Build failed! Check the output above.
    popd
    exit /b 1
)
popd
echo  Build succeeded.

REM --- Step 3: Find the JAR ----------------------------------------------
set "JAR_DIR=%MOD_DIR%build\libs"
set "JAR_FILE="
for %%f in ("%JAR_DIR%\data-harvester-*.jar") do (
    echo %%~nf | findstr /v "sources" >nul 2>&1
    if not errorlevel 1 (
        set "JAR_FILE=%%f"
    )
)

if "%JAR_FILE%"=="" (
    echo  ERROR: No JAR file found in %JAR_DIR%
    exit /b 1
)
echo  Built: %JAR_FILE%

REM --- Step 4: Install to Modrinth profile --------------------------------
if not exist "%MODRINTH_MODS%" (
    echo.
    echo  WARNING: Modrinth mods folder not found:
    echo    %MODRINTH_MODS%
    echo.
    echo  The JAR is at: %JAR_FILE%
    echo  Copy it to your Minecraft client mods folder manually.
    exit /b 0
)

REM Remove old versions first
del /q "%MODRINTH_MODS%\data-harvester-*.jar" 2>nul

copy "%JAR_FILE%" "%MODRINTH_MODS%\" >nul
echo.
echo  Installed to: %MODRINTH_MODS%
echo.
echo  ================================================
echo   DONE — Launch the Modrinth profile to use it
echo  ================================================
echo.

exit /b 0
