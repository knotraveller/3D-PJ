if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    return
}

$vsDevCmd = "E:\software\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "vcvars64.bat was not found at: $vsDevCmd"
}

$command = "`"$vsDevCmd`" >nul && set"
$environment = cmd.exe /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize the Visual Studio C++ build environment."
}

foreach ($line in $environment) {
    $separator = $line.IndexOf("=")
    if ($separator -le 0) {
        continue
    }

    $name = $line.Substring(0, $separator)
    $value = $line.Substring($separator + 1)
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

where.exe cl | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio environment was loaded, but cl.exe is still not available on PATH."
}
